import numpy as np  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  
import os
from openai import AzureOpenAI
import json
import requests
import time
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import argparse

# Configuration
endpoint = 'YOUR_AZURE_OPENAI_ENDPOINT'
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)
deployment_name = 'gpt-4o'

client = AzureOpenAI(
    azure_ad_token_provider=token_provider,
    azure_endpoint=endpoint,
    api_version='2024-02-01'
)

def call_gpt4_1(prompt):
    output = None
    message_text = [
                        {"role":"system","content":"You are an AI assistant that helps people find information."},
                        {"role":"user","content":prompt}
                    ]
    while output is None:
        try:
            time.sleep(0.5)
            completion = client.chat.completions.create(
                model = deployment_name,
                messages = message_text,
            )
            output = completion.choices[0].message.content
        except Exception as e:  
            print("API call exception:", e)  
            output = None
    return output

def cad_basic(args):
    with open(args.in_path, 'r', encoding='utf-8') as file:  
        data = json.load(file)    
        for idx, item in enumerate(data):  
            instruction = item['instruction']
            for _ in range(5):
                output = None
                output_json = None
                time.sleep(0.5)
                prompt = f"""## Task
You are a senior Computer-Aided Design (CAD) engineer. Your task is to provide clear, natural language editing instructions to a junior CAD designer for editing a sketch-and-extrude CAD model. Focus on geometric properties, including:
- The type and number of elements.
- Proportions and dimensions.
- Positional relationships between elements.
- Any other notable details.
## Sketch-and-Extrude Model Overview
An "extruded-sketch" is a 3D volume, formed by extruding a sketch. A "sketch-and-extrude" model is formed by multiple extruded-sketches via Boolean operations (i.e., add, cut, and intersect).
# Sketch
- A "sketch" is formed by one or multiple faces.
- A "face" is a 2D area bounded by loops.
- A "loop" is a closed path, consisting of one (i.e., circle) or multiple curves (e.g., line-arc-line).
- A "curve" (i.e., line, arc, or circle) is the lowest-level primitive.
  - A circle is defined by four points. 
  - An arc is defined by three points but with two points, where the third point is specified by the next curve (or the first curve when a loop is closed). 
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

## Your task
Original CAD Command Sequence:  
{item['original_sequence']}
Instruction: 
{instruction}
Your output should be of the following json format:
{{
    "modified sequence": your Modified CAD Command Sequence here.
}}
"""

                while output is None:
                    try:
                        output = call_gpt4_1(prompt)
                        try:
                            output_json = json.loads(output)
                        except:
                            output_json_lines = output.strip().splitlines()[1:-1]  
                            output_json = "\n".join(output_json_lines).strip()  
                            output_json = json.loads(output_json)
                    except Exception as e:
                        print("error: ", e)
                        time.sleep(1)
                        output = None
                
                item['output'] = output_json.get("modified sequence", None)
                
                with open(args.out_path, 'a', encoding='utf-8') as f:
                    json.dump(item, f, ensure_ascii=False, indent=4)
                    f.write(',\n') 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()
    cad_basic(args)