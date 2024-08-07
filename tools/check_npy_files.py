import os
import numpy as np

folder_path = '/home/Dataset/Data_prism_100/Data_400700/CAVE_ch166_expand'

npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

all_arrays = []

for npy_file in npy_files:
    file_path = os.path.join(folder_path, npy_file)
    data = np.load(file_path)
    all_arrays.append(data)

merged_array = np.concatenate(all_arrays, axis=0)

print(f'Merged array shape: {merged_array.shape}')
print(f'Total number of elements:{merged_array.size}')
