import json
import transformers
import torch
import argparse

def sample(args):
    model_path = "meta-llama/Meta-Llama-3-70B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    with open(args.in_path, 'r', encoding='utf-8') as file:  
        data = json.load(file)   
        
    for idx, item in enumerate(data):
        prompt = f"""## Task
You are a senior CAD engineer. Your task is to provide a clear and concise editing instruction (10 words or fewer) for editing a sketch-and-extrude CAD model. Your response should include:
1. Description of the Original CAD Model: Analyze the CAD operation sequence and describe the resulting geometry. Include element types (e.g., cylinder, prism, hole), quantities, proportions, spatial relationships, and any notable details.
2. Description of the Edited CAD Model: Analyze the CAD operation sequence and describe the resulting geometry. Include element types (e.g., cylinder, prism, hole), quantities, proportions, spatial relationships, and any notable details.
3. Change Analysis:  
    - Geometric Changes: Describe added, removed, or modified elements, including types (e.g., cylinder, prism, hole) and quantities (e.g., two rectangles). Use spatial or geometric features (e.g., "upper triangular face", "smaller rectangular prism", "central circular hole") instead of unintuitive terms like "first" or "second."
    - Proportions and Dimensions: Note changes in size, scaling, or relative proportions.
    - Positional Relationships: Explain spatial alignment and relationships between elements.
    - Other Notable Details: Highlight any additional observations.
    - Purpose: Suggest the intent behind the edit (e.g., "add a central hole", "remove the smaller prism", or "increase length by 8 units").
4. Editing Instruction: Provide a concise instruction (max 10 words) describing the modification.

## Sketch-and-Extrude Model Overview
An "extruded-sketch" is a 3D volume, formed by extruding a sketch. A "sketch-and-extrude" model is formed by multiple extruded-sketches via Boolean operations (i.e., add, cut, and intersect).
# Sketch
- A "sketch" is formed by one or multiple faces.
- A "face" is a 2D area bounded by loops.
- A "loop" is a closed path, consisting of one (i.e., circle) or multiple curves (e.g., line-arc-line).
- A "curve" (i.e., line, arc, or circle) is the lowest-level primitive.
    - A circle is defined by four points. 
    - An arc is defined by two points, with the third point specified by the next curve (or the first curve when a loop is closed). 
    - A line is defined by start point. 
- A point is represented by two integers which stands for the x and y coordinate, respectively.
- A loop with a circle can not contain additional curves since it is already a closed path. 
- When a face consists of multiple loops, the first loop defines the external boundary, and the remaining loops define internal loops (i.e., holes).
- An end-primitive token appears at the end of each primitive (curve, line, face, loop or sketch).
# Extrude
Each sketch will be followed by an extrude, which is represented by 18 parameters: BVVTTTRRRRRRRRRSOO
- B represents one of the three Boolean operations: add, cut or intersect. It occupies 1 parameter.
- V indicates the displacements of the top and the bottom planes from the reference plane in which a sketch is extruded to form a solid. It occupies 2 parameters.
- T represents 3D translation applied to the extruded solid. It occupies 3 parameters.
- R represents 3D rotation of the extrusion direction. It occupies 9 parameters.
- S represents the uniform scaling factor. It occupies 1 parameter.
- O represents the center of scaling as a 2D coordinate. It occupies 2 parameters.
# Note
- Note that every number is an integer.

## Your Task
Original CAD Sequence:
{item['original_sequence']}
Edited CAD Sequence:
{item['edited_sequence']}
Let's think step by step. Your output should be of the following json format:
{{
    "Description of the Original CAD Model": your description here.
    "Description of the Edited CAD Model": your description here.
    "Change Analysis": your change analysis here.
    "Editing Instruction": the final editing instruction here (10 words maximum).
}}
"""
        messages = [
            {"role": "system", "content": "You are an assistant trained to evaluate the semantic relevance between a Query and a Title. "},
            {"role": "user", "content": prompt},
        ]

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            batch_size=512
        )

        with open(f'{args.out_path}', 'a', encoding='utf-8') as f:
            item['instruction'] = outputs[0]["generated_text"][-1]['content']
            item['method'] = "sequence"
            f.write(json.dumps(item, ensure_ascii=False) + ",\n")

                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()
    sample(args)