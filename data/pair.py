import json
from collections import defaultdict
from itertools import combinations
import argparse

def pair_cad(args):
    with open(args.in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group items by the first 8 characters of their name
    groups = defaultdict(list)
    for item in data:
        key = item["name"][:8]
        groups[key].append(item)

    result = []

    for key, items in groups.items():
        if len(items) >= 2:
            temp_result = []

            for item1, item2 in combinations(items, 2):
                # Determine the type of change
                def determine_type(original_name, edited_name):
                    if "origInput" in original_name:
                        return "add"
                    elif "origInput" in edited_name:
                        return "delete"
                    else:
                        return "modify"

                type1 = determine_type(item1["name"], item2["name"])
                type2 = determine_type(item2["name"], item1["name"])

                # Forward combination
                temp_result.append({
                    "original_pic_name": item1["name"],
                    "edited_pic_name": item2["name"],
                    "original_sequence": item1["original_sequence"],
                    "edited_sequence": item2["original_sequence"],
                    "type": type1
                })

                # Reverse combination
                temp_result.append({
                    "original_pic_name": item2["name"],
                    "edited_pic_name": item1["name"],
                    "original_sequence": item2["original_sequence"],
                    "edited_sequence": item1["original_sequence"],
                    "type": type2
                })

                # Stop if 56 results have been generated for this group
                if len(temp_result) >= 56:
                    break

            result.extend(temp_result)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--out_path", type=str, required=True, help="Path to the output JSON file")
    args = parser.parse_args()

    pair_cad(args)
