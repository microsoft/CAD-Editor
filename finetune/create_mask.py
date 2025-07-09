import re
from difflib import SequenceMatcher
from copy import deepcopy
import json
import argparse

def parse_token(token):
    """
    Parse a token into its components.
    """
    return token.split() # split by space

def merge_consecutive_masks(tokens):
    """
    Merge consecutive <mask> tokens into a single <mask>.
    """
    merged_tokens = []
    for token in tokens:
        if token == "<mask>":
            if not merged_tokens or merged_tokens[-1] != "<mask>":
                merged_tokens.append("<mask>")
        else:
            merged_tokens.append(token)
    return merged_tokens

def compare_tokens(token1, token2):
    """
    Compare two tokens at a finer granularity.
    If they are partially similar (e.g., 'line,14,14' vs 'line,13,13'),
    preserve the common part and mask only the differences.
    """
    # Parse tokens into components
    components1 = parse_token(token1)
    components2 = parse_token(token2)
    
    # If the base command (e.g., 'line') is different, mask the entire token
    if components1[0] != components2[0]:
        return "<mask>"
    
    # If the base command is the same, compare the rest of the components
    result = [components1[0]]  # Start with the base command
    for comp1, comp2 in zip(components1[1:], components2[1:]):
        if comp1 == comp2:
            result.append(comp1)
        else:
            result.append("<mask>")
    
    # Reconstruct the token
    return ','.join(result)

def generate_mask_lcs_with_partial_matching(original_sequence, edited_sequence):
    """
    Generate the masked sequence using LCS with partial matching for tokens.
    """
    original_tokens = original_sequence.split()
    edited_tokens = edited_sequence.split()
    
    # Find the longest common subsequence (LCS)
    matcher = SequenceMatcher(None, original_tokens, edited_tokens)
    lcs = matcher.get_matching_blocks()  # Get matching blocks of tokens
    
    masked_sequence = []
    original_idx = 0
    edited_idx = 0

    for match in lcs:
        # Handle tokens in original_sequence that are not in LCS (deletions)
        while original_idx < match.a:
            masked_sequence.append("<mask>")
            original_idx += 1

        # Handle tokens in edited_sequence that are not in LCS (additions)
        while edited_idx < match.b:
            masked_sequence.append("<mask>")
            edited_idx += 1

        # Add the matching tokens (LCS part), with partial matching for differences
        for i in range(match.size):
            token1 = original_tokens[original_idx]
            token2 = edited_tokens[edited_idx]
            if token1 == token2:
                masked_sequence.append(token1)
            else:
                masked_sequence.append(compare_tokens(token1, token2))
            original_idx += 1
            edited_idx += 1

    # Handle remaining tokens in original_sequence (deletions)
    while original_idx < len(original_tokens):
        masked_sequence.append("<mask>")
        original_idx += 1

    # Handle remaining tokens in edited_sequence (additions)
    while edited_idx < len(edited_tokens):
        masked_sequence.append("<mask>")
        edited_idx += 1

    # Merge consecutive <mask> tokens
    merged_sequence = merge_consecutive_masks(masked_sequence)
    return " ".join(merged_sequence)

def process_dataset_with_partial_matching(dataset):
    """
    Process the dataset to add `original_sequence_mask` for each entry using partial matching.
    """
    processed_data = []
    
    for entry in dataset:
        # Skip entries without both original_sequence and edited_sequence
        if 'original_sequence' not in entry or 'edited_sequence' not in entry:
            continue
        
        original_sequence = entry['original_sequence']
        edited_sequence = entry['edited_sequence']
        
        # Generate the masked sequence using partial matching
        original_sequence_mask = generate_mask_lcs_with_partial_matching(original_sequence, edited_sequence)
        
        # Add the masked sequence to the entry
        new_entry = deepcopy(entry)
        new_entry['masked_sequence'] = original_sequence_mask
        processed_data.append(new_entry)
    
    return processed_data


def main(args):
    # load and process a JSON dataset
    with open(args.input_path, "r") as f:
        dataset = json.load(f)

    updated_dataset = process_dataset_with_partial_matching(dataset)

    with open(args.output_path, "w") as f:
        json.dump(updated_dataset, f, indent=4)
    
    print(f"Processed {len(updated_dataset)} entries from {args.input_path} and saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset to create masked sequences')
    parser.add_argument('--input_path', type=str, default='raw_train.json', 
                        help='Path to input JSON dataset file (default: raw_train.json)')
    parser.add_argument('--output_path', type=str, default='train.json',
                        help='Path to output JSON dataset file (default: train.json)')
    
    args = parser.parse_args()

    main(args)


