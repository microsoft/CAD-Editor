import os
import argparse
from data import SE
import pickle

NUM_TRHEADS = 36
NUM_FOLDERS = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input folder of the CAD obj (after normalization)")
    parser.add_argument("--bit", type=int, required=True, help='Number of bits for quantization')
    parser.add_argument("--output", type=str, required=True, help="Output file path to save the data")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Start creating dataset 
    parser = SE(start=0, end=NUM_FOLDERS, datapath=args.input, bit=args.bit, threads=NUM_TRHEADS) # number of threads in your pc
    train_samples = parser.load_all_obj()

    # Save to file 
    with open(args.output, "wb") as tf:
        pickle.dump(train_samples, tf) 