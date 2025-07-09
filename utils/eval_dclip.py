import torch  
from torchvision import transforms  
from PIL import Image  
from directional_clip_score import CLIPLoss
import json
import os
from collections import defaultdict
import heapq
import argparse

def preprocess_image(pil_image):
    # Define a transform to convert the PIL image to a tensor  
    transform = transforms.Compose([  
        transforms.ToTensor(),  # Converts a PIL Image or numpy.ndarray (H x W x C) to a FloatTensor of shape (C x H x W)  
    ])  
    
    # Apply the transform to the PIL image  
    tensor_image = transform(pil_image)  
    return tensor_image


def find_files_with_prefix_source(directory, prefix):
    files = sorted(os.listdir(directory))

    matching_files = [
        file_name for file_name in files 
        if file_name[5:].startswith(prefix)
    ]
    return matching_files


def find_files_with_prefix(directory, prefix):
    files = sorted(os.listdir(directory))

    matching_files = [
        file_name for file_name in files 
        if file_name.startswith(prefix)
    ]
    return matching_files


def cal_dclip(args):
    source_files = sorted(os.listdir(args.source_dir)) 
    
    with open(args.instruction_path, 'rb') as file:  
        ins_data = json.load(file)  
    
    # Collect all DCLIP scores for average calculation
    all_dclip_scores = []
    total_comparisons = 0
    
    for idx, item in enumerate(ins_data):
        prefix_source = str(idx).zfill(5) + '_' 
        prefix_edit = str(idx).zfill(5) + '_'
        
        source_files = find_files_with_prefix(args.source_dir, prefix_source)
        matching_files = find_files_with_prefix(args.edit_dir, prefix_edit) 
        print(prefix_edit, matching_files)
        
        for source_file in source_files:
            source_path = os.path.join(args.source_dir, source_file)
            for matching_file in matching_files:
                matching_path = os.path.join(args.edit_dir, matching_file)  
                print(matching_path)
                
                try:
                    src_img = Image.open(source_path)  
                    target_img = Image.open(matching_path)  
                    src_img = preprocess_image(src_img)
                    target_img = preprocess_image(target_img)

                    device = "cuda" if torch.cuda.is_available() else "cpu"  
                    
                    clip_loss_module = CLIPLoss(device=device)  
                    
                    src_img_tensor = clip_loss_module.preprocess(src_img).unsqueeze(0).to(device)  
                    target_img_tensor = clip_loss_module.preprocess(target_img).unsqueeze(0).to(device)  
                    
                    src_text = "This is a 3D shape. "
                    instruction = item['instruction']
                    target_text = src_text + instruction

                    directional_loss = clip_loss_module.clip_directional_loss(src_img_tensor, src_text, target_img_tensor, target_text)  
                    dclip_score = directional_loss.item()

                    # Collect score for average calculation
                    all_dclip_scores.append(dclip_score)
                    total_comparisons += 1

                    item['dclip'] = dclip_score

                    # Write the results as a JSON list
                    with open(args.out_path, 'a', encoding='utf-8') as f:
                        json.dump(item, f, ensure_ascii=False)
                        f.write(',\n')
                                
                except Exception as e:
                    print(f"Error processing {source_path} -> {matching_path}: {e}")
                    continue

    # Calculate and print statistics
    if all_dclip_scores:
        average_dclip = sum(all_dclip_scores) / len(all_dclip_scores)
        max_dclip = max(all_dclip_scores)
        min_dclip = min(all_dclip_scores)
        
        print(f"\n{'='*50}")
        print(f"DCLIP Score Statistics:")
        print(f"{'='*50}")
        print(f"Total comparisons: {total_comparisons}")
        print(f"Average DCLIP score: {average_dclip:.6f}")
        print(f"Maximum DCLIP score: {max_dclip:.6f}")
        print(f"Minimum DCLIP score: {min_dclip:.6f}")
        print(f"{'='*50}")
        
    else:
        print("No DCLIP scores were calculated!")
    
    return all_dclip_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--edit_dir", type=str, required=True)
    parser.add_argument("--instruction_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="cad_dclip.json")

    args = parser.parse_args()
    
    cal_dclip(args)