import json
import argparse

def merge_json_files(args):
    """
    Merge two JSON files
    """
    with open(args.file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    with open(args.file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    merged_data = data1 + data2
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"Merge completed!")
    print(f"File 1: {len(data1)} items")
    print(f"File 2: {len(data2)} items")
    print(f"Merged: {len(merged_data)} items")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two JSON files")
    parser.add_argument("--file1", type=str, required=True, help="Path to first JSON file")
    parser.add_argument("--file2", type=str, required=True, help="Path to second JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    
    args = parser.parse_args()
    merge_json_files(args) 