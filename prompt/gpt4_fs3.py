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
            print("API call error:", e)  
            output = None
    return output

def cad_fs3(args):
    with open(args.example_path, 'r', encoding='utf-8') as file:  
        example_data = json.load(file)   

    with open(args.in_path, 'r', encoding='utf-8') as file:  
        data = json.load(file)    
        for idx, item in enumerate(data):  
            for _ in range(5):
                output = None
                output_json = None
                time.sleep(1)
                descriptions = [entry['instruction'] for entry in example_data]
                vectorizer = TfidfVectorizer()  
                tfidf_matrix = vectorizer.fit_transform(descriptions + [item['instruction']])  
                cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])  
                top_indices = np.argsort(cosine_similarities[0])[::-1][:6]  
                top_entries = [example_data[index] for index in top_indices]  

                prompt = f"""Modify the original Computer-Aided Design(CAD) command sequence according to the instruction:\n'

## Instructions for sketch-and-extrude model
A sketch-and-extrude model consists of multiple extruded-sketches.
# Sketch
- A sketch consists of multiple faces
- A face consists of multiple loops.
- A loop consists of multiple curves.
- A curve is either a line, an arc, or a circle.
- A circle is defined by four points with four geometry tokens. 
- An arc is defined by three points but with two tokens, where the third point is specified by the next curve (or the first curve when a loop is closed). 
- A line is defined by start point. 
- A point is represented by two integers which stands for the x and y coordinate, respectively.
- A loop with a circle can not contain additional curves since it is already a closed path. 
- When a face consists of multiple loops, the first loop defines the external boundary, and the remaining loops define internal loops (i.e., holes).
- An end-primitive token appears at the end of each primitive (curve, line, face, loop or sketch).
# Extrude
Each sketch will be followd by an extrude, which is represented by 18 parameters: BWVTTTRRRRRRRRRSOO
- B represents one of the three Boolean operations: add, cut or intersect. It occupies 1 parameter
- V indicates the displacements of the top and the bottom planes from the referenceplane in which a sketch is extruded to form a solid. It occupies 2 parameters.T represents 3D translation applied to the extruded solid. It occupies 3parameters
- R represents 3D rotation of the extrusion direction. It occupies 6 parameters.
- S represents the uniform scaling factor. It occupies 1 parameter.
- O represents the center of scaling as a 2D coordinate. It occupies 2 parameters.
# Note
- Note that every number is an integer.

## Examples for editing sketch-and-extrude model
Example 1:
Original CAD Command Sequence:
{top_entries[0]['original_sequence']}
Instruction: 
{top_entries[0]['instruction']}
Modified CAD Command Sequence:
{top_entries[0]['edited_sequence']}
Example 2:
Original CAD Command Sequence:  
{top_entries[1]['original_sequence']}
Instruction: 
{top_entries[1]['instruction']}
Modified CAD Command Sequence:
{top_entries[1]['edited_sequence']}
Example 3:
Original CAD Command Sequence:  
{top_entries[2]['original_sequence']}
Instruction: 
{top_entries[2]['instruction']}
Modified CAD Command Sequence:
{top_entries[2]['edited_sequence']}


## Your task
Original CAD Command Sequence:  
{item['original_sequence']}
Instruction: 
{item[instruction_field]}
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
    parser.add_argument("--example_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()
    cad_fs3(args)