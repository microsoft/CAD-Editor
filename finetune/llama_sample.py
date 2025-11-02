import argparse
import json
from tqdm import tqdm
import transformers
from peft import PeftModel
from pathlib import Path
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
MAX_LENGTH = 1024

def prepare_model_and_tokenizer(args):
    model_id= 'meta-llama/Meta-Llama-3-8B-Instruct'
    pipeline = transformers.pipeline("text2text-generation",
                                        model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map='auto')
    tokenizer = pipeline.tokenizer
    model = pipeline.model

    model.eval()

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=tokenizer,
        model=model,
    )

    model = PeftModel.from_pretrained(model, args.model_path, device_map="auto")
    # merge
    # model.merge_and_unload()
    
    return model, tokenizer


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    llama_tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_prompt_template(task_type, item, instruction):
    """Get prompt template based on task type"""
    if task_type == "mask":
        return f"""Below is a Computer-Aided Design (CAD) operation sequence, replace the parts that need to be modified with the string "<mask>" according to the editing instruction.
Original CAD Operation Sequence:
{item['original_sequence']}
Editing Instruction:
{instruction}
Masked CAD Operation Sequence:
"""
    elif task_type == "infill":
        return f"""Below is the original Computer-Aided Design (CAD) operation sequence. 
Original CAD Operation Sequence:
{item['original_sequence']}

The parts that need to be modified according to the editing instruction have been replaced by the string "<mask>".
Editing Instruction:
{instruction}
Masked CAD Operation Sequence:
{item['output_mask']}

Based on the original CAD sequence, the editing instruction, and the masked sequence, generate the complete edited CAD sequence by replacing "<mask>" with the appropriate content:
"""
    else:
        raise ValueError(f"Unknown task: {task_type}")


def get_output_field_name(task_type):
    """Get output field name based on task type"""
    if task_type == "mask":
        return "output_mask"
    elif task_type == "infill":
        return "output_infill"
    else:
        raise ValueError(f"Unknown task: {task_type}")


def conditional_sample(args):
    model, tokenizer = prepare_model_and_tokenizer(args)
    with open(args.data_path, 'r', encoding='utf-8') as file: 
        content = file.read().strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            try:
                data =[]
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse JSON from {args.data_path}. Please check the file format.")

    output_field = get_output_field_name(args.task_type)
    
    for idx, item in enumerate(data):
        instruction = item['instruction']
        
        # Check if infill task requires output_mask field
        if args.task_type == "infill" and 'output_mask' not in item:
            print(f"Warning: Item {idx} missing 'output_mask' field required for infill task. Skipping.")
            continue
        
        prompts = []
        for _ in range(args.num_samples):
            prompt = get_prompt_template(args.task_type, item, instruction)
            prompts.append(prompt)

        outputs = []

        while len(outputs) < args.num_samples:
            batch_prompts = prompts[len(outputs) : len(outputs) + args.batch_size]

            batch = tokenizer(
                list(batch_prompts),
                return_tensors="pt",
            )
            batch = {k: v.cuda() for k, v in batch.items()}

            generate_ids = model.generate(
                **batch,
                do_sample=True,
                max_new_tokens=MAX_LENGTH,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.3,
            )
           
            gen_strs = tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            outputs.extend(gen_strs)
            print(f"Generated {len(outputs)}/{args.num_samples} samples.")
        
        with open(args.out_path, "a+") as f:
            for prompt, output in zip(prompts, outputs):
                item[output_field] = output[len(prompt):]
                f.write(json.dumps(item) + "\n")
                            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, required=True, choices=["mask", "infill"], 
                       help="Task to perform: 'mask' for masking CAD sequences, 'infill' for infilling from masked sequences")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size (default: 32)")
    parser.add_argument("--out_path", type=str, default="cad_samples.jsonl")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    print(f"Running {args.task_type} task with batch size {args.batch_size}")
    conditional_sample(args) 