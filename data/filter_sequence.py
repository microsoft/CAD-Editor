import json
import argparse

def filter(args):
    with open(args.in_path, 'r') as file:  
        data = json.load(file)
    
    filtered_items = []  
    
    for item in data:  
        original_sequence = item['original_sequence']
        edited_sequence = item['edited_sequence']  
    
        original_extrude_count = original_sequence.count('<extrude_end>')
        
        edited_extrude_count = edited_sequence.count('<extrude_end>')
        
        # Check if an instruction contains 3 or more edits
        instruction = item['instruction'].lower()
        punctuation_count = instruction.count(',') + instruction.count(';')  
        
        wordlist = ["no transformation", "are identical"]
        # Check if any word in wordlist is in the instruction
        contains_wordlist = any(word in instruction for word in wordlist)
        
        if (original_extrude_count <= 3 and edited_extrude_count <= 3 and
            punctuation_count <= 2 and not contains_wordlist):  
            filtered_items.append(item)  
    
    with open(args.out_path, 'w') as file:  
        json.dump(filtered_items, file, indent=4)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()
    
    filter(args)
