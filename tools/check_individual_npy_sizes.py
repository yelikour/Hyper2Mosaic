import os
import numpy as np

folder_path = '/home/Dataset/Data_prism_100/Data_400700/CAVE_ch166_expand'

npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])

for npy_file in npy_files:
    file_path = os.path.join(folder_path, npy_file)
    data = np.load(file_path)
    print(f'File: {npy_file}, Shape: {data.shape}, Size: {data.size}, dtype: {data.dtype}, Memory Size (bytes): {data.nbytes}')
