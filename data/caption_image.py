import os
import requests
import base64
import json
import time
import argparse
from mimetypes import guess_type
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


def local_image_to_data_url(image_path):
    # Encode a local image into data URL
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream' 
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

# Configuration - IMPORTANT: Update these values before running
# Replace with your actual Azure OpenAI endpoint
endpoint = 'https://your-azure-openai-endpoint.openai.azure.com/'
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

def call_gpt4o_1(prompt, image_path):
    output = None
    message_text = [
                        {"role":"system","content":"You are an AI assistant that helps people find information."},
                        {"role":"user","content":[
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(image_path)}
                            }
                        ]}
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
            print("API call exceptions:", e)  
            output = None
    return output

def call_gpt4o_2(prompt1, image_path1, output1, prompt2, image_path2):
    output = None
    message_text = [
                        {"role":"system","content":"You are an AI assistant that helps people find information."},
                        {"role":"user","content":[
                            {
                                "type": "text",
                                "text": prompt1
                            },
                            {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(image_path1)}
                            }
                        ]},
                        {"role":"assistant","content":output1},
                        {"role":"user","content":[
                            {
                                "type": "text",
                                "text": prompt2
                            },
                            {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(image_path2)}
                            }
                        ]}
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
            print("API call exceptions:", e)  
            time.sleep(1)
            output = None
    return output


def call_gpt4o_3(prompt1, image_path1, output1, prompt2, image_path2, output2, prompt3):
    output = None
    message_text = [
                        {"role":"system","content":"You are an AI assistant that helps people find information."},
                        {"role":"user","content":[
                            {
                                "type": "text",
                                "text": prompt1
                            },
                            {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(image_path1)}
                            }
                        ]},
                        {"role":"assistant","content":output1},
                        {"role":"user","content":[
                            {
                                "type": "text",
                                "text": prompt2
                            },
                            {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(image_path2)}
                            }
                        ]},
                        {"role":"assistant","content":output2},
                        {"role":"user","content":prompt3}
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
            print("API call exceptions:", e)  
            time.sleep(1)
            output = None
    return output


def call_gpt4o_4(prompt1, image_path1, output1, prompt2, image_path2, output2, prompt3, output3, prompt4):
    output = None
    message_text = [
                        {"role":"system","content":"You are an AI assistant that helps people find information."},
                        {"role":"user","content":[
                            {
                                "type": "text",
                                "text": prompt1
                            },
                            {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(image_path1)}
                            }
                        ]},
                        {"role":"assistant","content":output1},
                        {"role":"user","content":[
                            {
                                "type": "text",
                                "text": prompt2
                            },
                            {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(image_path2)}
                            }
                        ]},
                        {"role":"assistant","content":output2},
                        {"role":"user","content":prompt3},
                        {"role":"assistant","content":output3},
                        {"role":"user","content":prompt4}
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
            print("API call exceptions:", e)  
            time.sleep(1)
            output = None
    return output


def find_files_with_prefix(directory, prefix):  
    files = sorted(os.listdir(directory))  
    matching_files = [file_name for file_name in files if file_name.startswith(prefix)]  

    return matching_files  

def multi_level_captioning(args):
    with open(args.sequence_dir, 'r', encoding='utf-8') as file:  
        data = json.load(file)   

    for idx, item in enumerate(data):
        original_prefix = item['original_pic_name']+'_'
        original_file = find_files_with_prefix(args.image_dir, original_prefix)[0]
        source_path = os.path.join(args.image_dir, original_file)
        edit_prefix = item['edited_pic_name']+'_'
        edit_file = find_files_with_prefix(args.image_dir, edit_prefix)[0]
        edit_path = os.path.join(args.image_dir, edit_file)

        print(original_file, edit_file)
        
        time.sleep(0.5)
        output1 = None
        output2 = None
        output3 = None
        output4 = None

        prompt1 = """Please take a look at the first of two 3D shapes we'll be examining. Please provide a detailed description, focusing on its geometric properties, including the type and number of elements it features, the proportions of its size, its positional relationships between elements, and any additional details that stand out."""
        prompt2 = """Now, let's turn our attention to the second 3D shape. Please provide a detailed description, focusing on its geometric properties, including the type and number of elements it features, the proportions of its size, its positional relationships between elements, and any additional details that stand out."""
        prompt3 = """Please provide concise instructions for transforming the first 3D shape into the second. """
        prompt4 = """Condense your instructions to one sentence, 10 words maximum."""    
        
        while output1 is None or str(output1).startswith("I'm sorry"):
            try:
                output1 = call_gpt4o_1(prompt1, source_path)
            except requests.RequestException as e:  
                print(f"Request failed: {e}")
                time.sleep(0.5)  
                output1 = None  
        while output2 is None or str(output2).startswith("I'm sorry"):
            try:
                output2 = call_gpt4o_2(prompt1, source_path, output1, prompt2, edit_path)
            except requests.RequestException as e:  
                print(f"Request failed: {e}")
                time.sleep(0.5)  
                output2 = None  
        while output3 is None or str(output3).startswith("I'm sorry"):
            try:
                output3 = call_gpt4o_3(prompt1, source_path, output1, prompt2, edit_path, output2, prompt3)
            except requests.RequestException as e:  
                print(f"Request failed: {e}")
                time.sleep(0.5)  
                output3 = None  
        while output4 is None or str(output4).startswith("I'm sorry"):
            try:
                output4 = call_gpt4o_4(prompt1, source_path, output1, prompt2, edit_path, output2, prompt3, output3, prompt4)
            except requests.RequestException as e:  
                print(f"Request failed: {e}")
                time.sleep(0.5)  
                output4 = None  

        result = {
            "original_pic_name": original_file,
            "edited_pic_name": edit_file,
            "original_sequence": item['original_sequence'],
            "edited_sequence": item['edited_sequence'],
            "type": item["type"],
            "method": "image",
            "description_original": output1,
            "description_edited":output2,
            "detailed_instruction": output3,
            "instruction":output4,
        }

        with open(args.caption_path, 'a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
            f.write(',\n') 
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--caption_path", type=str, required=True)
    args = parser.parse_args()
    
    multi_level_captioning(args)