from util import process_obj_se
from tqdm import tqdm
from multiprocessing import Pool
import json 
from pathlib import Path
from glob import glob
import itertools
 

class SE():
    """ sketch-extrude dataset """
    def __init__(self, start, end, datapath, bit, threads=16):
        self.start = start
        self.end = end
        self.datapath = datapath
        self.threads = threads
        self.bit = bit

    def load_all_obj(self):
        print("Loading obj data...")

        # with open('../data/train_val_test_split.json') as f:
        #     data_split = json.load(f)
       
        project_folders = []
        cur_dir =  Path(self.datapath)
        # print(cur_dir)
        project_folders += glob(str(cur_dir)+'/*/')
        print(project_folders)
        # Parallel loader
        iter_data = zip(
            project_folders,
            itertools.repeat(self.bit),
        )
        samples = []
        load_iter = Pool(self.threads).imap(process_obj_se, iter_data)
        for data_sample in tqdm(load_iter, total=len(project_folders)):
            samples += data_sample
        
        print('Splitting data...')
        train_samples = []
 
        for data in tqdm(samples):
            train_samples.append(data) # put into training if no match

        print(f"Data Summary")
        print(f"\tTraining data: {len(train_samples)}")
        # print(f"\tValidation data: {len(val_samples)}")
        # print(f"\tTest data: {len(test_samples)}")
        return train_samples

