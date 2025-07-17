from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import transformers
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import numpy as np
import warnings
import random
import torch
import argparse
import glob
import os
# from diffusers.training_utils import cast_training_params


IGNORE_INDEX = -100
MAX_LENGTH = 1024  
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class CADDataset(Dataset):
    def __init__(self, json_fn, task_type="mask", llama_tokenizer=None):
        if not os.path.exists(json_fn):
            raise ValueError(f"{json_fn} does not exist")
        self.inputs = json.load(open(json_fn, "r"))
        self.llama_tokenizer = llama_tokenizer
        self.task_type = task_type  # 'mask', 'infill', or 'infill_selective'

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        item = self.inputs[index]
        src_seq = item['original_sequence']
        mask_seq = item['masked_sequence']
        instruction = item['instruction']
        
        # For infilly tasks, also get edited sequence
        if (self.task_type == "infill" or self.task_type == "infill_selective") and 'edited_sequence' in item:
            edit_seq = item['edited_sequence']
            val = self.tokenize(src_seq, instruction, mask_seq, edit_seq)
        else:
            val = self.tokenize(src_seq, instruction, mask_seq)
            
        return val

    def tokenize(self, src_seq, instruction, mask_seq, edit_seq=None):
        if self.task_type == "infill" or self.task_type == "infill_selective":
            tokens, prompt_length = self.conditional_generation_infill(
                src_seq, instruction, mask_seq, edit_seq)
        else:
            tokens, prompt_length = self.conditional_generation_mask(
                src_seq, instruction, mask_seq)
            
        input_ids = tokens.input_ids[0]
        labels = tokens.input_ids[0].clone()
        # Set the labels for the prompt part to IGNORE_INDEX so they are ignored in loss calculation
        labels[:prompt_length] = IGNORE_INDEX
        input_id_lens = label_lens = (
            tokens.input_ids.ne(self.llama_tokenizer.pad_token_id).sum().item()
        )
        return dict(
            input_ids=input_ids,
            input_id_lens=input_id_lens,
            labels=labels,
            label_lens=label_lens,
        )

    def conditional_generation_mask(self, src_seq, instruction, mask_seq):
        prompt = f"""Below is a Computer-Aided Design (CAD) operation sequence, replace the parts that need to be modified with the string "<mask>" according to the editing instruction.
Original CAD Operation Sequence:
{src_seq}
Editing Instruction:
{instruction}
Masked CAD Operation Sequence:
"""

        full_text = prompt + mask_seq + self.llama_tokenizer.eos_token
        tokens = self.llama_tokenizer(
            full_text,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            truncation=True,
        )
        prompt_length = len(self.llama_tokenizer(prompt)['input_ids'])
        return tokens, prompt_length

    def conditional_generation_infill(self, src_seq, instruction, mask_seq, edit_seq):
        prompt = f"""Below is the original Computer-Aided Design (CAD) operation sequence. 
Original CAD Operation Sequence:
{src_seq}

The parts that need to be modified according to the editing instruction have been replaced by the string "<mask>".
Editing Instruction:
{instruction}
Masked CAD Operation Sequence:
{mask_seq}

Based on the original CAD sequence, the editing instruction, and the masked sequence, generate the complete edited CAD sequence by replacing "<mask>" with the appropriate content:
"""

        full_text = prompt + edit_seq + self.llama_tokenizer.eos_token
        tokens = self.llama_tokenizer(
            full_text,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            truncation=True,
        )
        prompt_length = len(self.llama_tokenizer(prompt)['input_ids'])
        return tokens, prompt_length


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple(
            [instance[key].clone().detach() for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def setup_datasets(args, llama_tokenizer, transform_args={}):
    train_file = "train.json"
    val_file = "test.json"
    
    datasets = {
        "train": CADDataset(
            str(args.data_folder / train_file),
            task_type=args.task_type,
            llama_tokenizer=llama_tokenizer,
        ),
        "val": CADDataset(
            str(args.data_folder / val_file),
            task_type=args.task_type,
            llama_tokenizer=llama_tokenizer,
        ),
    }

    return datasets


def setup_training_args(args):
    output_dir = args.expdir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        report_to = "none"
    else:
        report_to = "wandb"
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    training_args = TrainingArguments(
        fsdp=False,
        fp16=not args.fp8,
        bf16=False,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        num_train_epochs=args.num_epochs,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=10,
        eval_strategy="steps",  # Use modern parameter name
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.num_warmup_steps,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=output_dir,
        run_name=args.run_name,
        report_to=report_to,
        dataloader_num_workers=8,
        remove_unused_columns=False,
        # this is just to get trainer to behave how I want
        label_names=["cad_ids"],
    )
    return training_args


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


def setup_model(args, rank):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipeline = transformers.pipeline("text2text-generation",
                                     model=model_id, model_kwargs={"torch_dtype": torch.bfloat16})
    llama_tokenizer = pipeline.tokenizer
    base_model = pipeline.model

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # For selective infilly, load a pre-trained infilly model checkpoint
    if args.task_type == "infill_selective" and args.pretrained_model_path:
        print(f"Loading pre-trained model from {args.pretrained_model_path}")
        peft_model = PeftModel.from_pretrained(base_model, args.pretrained_model_path, device_map="auto")
        peft_model.to(device)
        original_state_dict = {f"{k}": v for k, v in peft_model.state_dict().items()}
        
        model = get_peft_model(base_model, lora_config)
        model.load_state_dict(original_state_dict, strict=True)
    else:
        # For mask or initial infilly training
        model = get_peft_model(base_model, lora_config)
    
    model.print_trainable_parameters()

    special_tokens_dict = dict()
    if llama_tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if llama_tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if llama_tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if llama_tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=llama_tokenizer,
        model=model,
    )

    return model, llama_tokenizer


def setup_trainer(args):
    training_args = setup_training_args(args)
    model, llama_tokenizer = setup_model(args, training_args.local_rank)
    
    datasets = setup_datasets(args, llama_tokenizer)

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=llama_tokenizer,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=data_collator,
    )

    return trainer


def main(args):
    trainer = setup_trainer(args)

    if args.resume_dir is not None:
        train_result = trainer.train(resume_from_checkpoint=args.resume_dir)
    else:
        train_result = trainer.train()

    print(train_result)
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, choices=["mask", "infill", "infill_selective"], 
                        default="mask", help="Task type: 'mask' for masking parts, 'infill' for infilly training, 'infill_selective' for selective infilly training")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, default=None, 
                        help="Path to pretrained model checkpoint (required for infill_selective)")
    parser.add_argument("--expdir", type=Path, default="model/")
    parser.add_argument("--fp8", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--data_folder", type=Path, default="/data/dataset/")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eval_freq", default=100000000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)
    parser.add_argument("--resume_dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    # Validate arguments
    if args.task_type == "infill_selective" and not args.pretrained_model_path:
        raise ValueError("For infill_selective training, pretrained_model_path must be provided")
        
    # Set WANDB project name
    if not args.debug:
        os.environ["WANDB_PROJECT"] = "CAD-Editor"
    print(args)
    main(args) 