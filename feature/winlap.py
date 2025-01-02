import os
import pickle
import numpy as np
import pandas as pd
from natsort import natsorted
import glob
import csv
import argparse
from tqdm import tqdm
import parmap
import multiprocessing
from multiprocessing import Manager


manager = Manager()
X_y = manager.list()


# our feature: overlapping tam with size and sum concatenated
def winlap(instance, max_matrix_len=1800, max_time=80):
    times = []
    sizes = []
    for line in instance:
        time, size = line.split('\t')
        times.append(float(time))
        size = size.split('\n')[0]
        sizes.append(int(float(size)))
    
    window_size = max_time / max_matrix_len
    sliding_interval = window_size / 2
    max_matrix_len = max_matrix_len * 2

    feature = [0] * max_matrix_len * 2

    for i in range(len(sizes)):
        idx = int(times[i] / max_time * max_matrix_len)
        if idx >= max_matrix_len:
            idx = max_matrix_len - 1
        
        if sizes[i] < 0: # incoming
            feature[idx] += 1
            feature[idx + max_matrix_len] += -sizes[i]
            pre_idx = max(idx-1, 0)
            feature[pre_idx] += 1
            feature[pre_idx + max_matrix_len] += -sizes[i]

    return feature


def process_file(file, X_y):
    with open(file, 'r') as f:
        instance = f.readlines()
    index = file.split('/')[-1].split('_')[0]
    feature = winlap(instance)
    feature.append(int(index))
    X_y.append(feature)


def extract_features(input_folder, feature_func):
    files = natsorted(glob.glob(os.path.join(input_folder, '*.txt')))
    print(len(files))
    
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_processes) as pool:
        parmap.map(process_file, files, X_y, pm_pbar=True, pm_processes=num_processes)

    print("Saving to X and y...")
    X_y_array = np.array(X_y)
    X = X_y_array[:, :-1]
    y = X_y_array[:, -1].reshape(-1, 1)

    return X, y


def save_to_pickle(data, filename):
    print("Saving to pickle...")
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Features saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some folders and rule names.')
    parser.add_argument('--input-folder', type=str, default="/path/to/input_folder", help='Root folder for input features')
    parser.add_argument('--x-path', type=str, default="/path/to/x_train.pkl", help='Path to the X features file')
    parser.add_argument('--y-path', type=str, default="/path/to/y_train.pkl", help='Path to the Y features file')
    args = parser.parse_args()

    input_folder = args.input_folder
    X_path = args.x_path
    y_path = args.y_path
    
    max_matrix_len = 2000
    max_time = 60
    
    X, y = extract_features(input_folder, lambda instance: winlap(instance, max_matrix_len, max_time))

    save_to_pickle(y, y_path)
    save_to_pickle(X, X_path)
