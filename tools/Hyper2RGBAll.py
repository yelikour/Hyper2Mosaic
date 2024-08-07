# -*-coding: utf-8 -*-
# @Time    : 2024/8/7 17:00
# @Author  : YeLi
# @File    : Hyper2RGBAll.py
# @Software: PyCharm
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import interp1d

def load_cie_data(cie_path):
    cie_file_path = os.path.join(cie_path, 'CIE_xyz_1964_10deg.csv')
    cie_data = pd.read_csv(cie_file_path, delimiter=',', header=None)
    cie_data.fillna(0, inplace=True)
    if cie_data.shape[1] != 4:
        raise ValueError(f"Expected 4 columns, but got {cie_data.shape[1]}")
    cie_data.columns = ['wavelength', 'x_bar', 'y_bar', 'z_bar']
    return cie_data

def process_image(image_file, cie_data, use_d65, output_folder):
    hyperspectral_image = np.load(image_file)
    selected_wavelengths = np.arange(400, 701, 10)
    selected_indices = cie_data['wavelength'].isin(selected_wavelengths)

    wavelengths = cie_data.loc[selected_indices, 'wavelength'].values
    x_bar = cie_data.loc[selected_indices, 'x_bar'].values
    y_bar = cie_data.loc[selected_indices, 'y_bar'].values
    z_bar = cie_data.loc[selected_indices, 'z_bar'].values

    print(f"Selected wavelengths: {len(wavelengths)}")
    print(f"Hyperspectral image shape: {hyperspectral_image.shape[2]}")

    assert len(wavelengths) == hyperspectral_image.shape[2], "Number of wavelengths and hyperspectral image slices do not match."

    if use_d65:
        d65_file_path = '../Dataset/CIE/CIE_std_illum_D65.csv'
        d65_data = pd.read_csv(d65_file_path, delimiter=',', header=None)
        d65_data.fillna(0, inplace=True)
        d65_data.columns = ['wavelength', 'intensity']
        d65_selected = d65_data.loc[d65_data['wavelength'].isin(selected_wavelengths), 'intensity'].values
    else:
        d65_selected = np.ones(len(wavelengths))

    height, width = hyperspectral_image.shape[:2]
    X = np.zeros((height, width))
    Y = np.zeros((height, width))
    Z = np.zeros((height, width))

    for i in range(len(wavelengths)):
        X += hyperspectral_image[:, :, i] * x_bar[i] * d65_selected[i]
        Y += hyperspectral_image[:, :, i] * y_bar[i] * d65_selected[i]
        Z += hyperspectral_image[:, :, i] * z_bar[i] * d65_selected[i]

    X /= np.max(X)
    Y /= np.max(Y)
    Z /= np.max(Z)

    XYZ_to_sRGB_matrix = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])

    RGB_image = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            XYZ = np.array([X[i, j], Y[i, j], Z[i, j]])
            RGB = np.dot(XYZ_to_sRGB_matrix, XYZ)
            RGB_image[i, j, :] = RGB

    RGB_image = np.clip(RGB_image, 0, 1)

    output_file = os.path.join(output_folder, os.path.basename(image_file).replace('.npy', '.png'))
    plt.imsave(output_file, RGB_image)

def main():
    parser = argparse.ArgumentParser(description="Process hyperspectral images to RGB")
    parser.add_argument('--dataset_folder', type=str, required=True, help="Path to the dataset folder containing the four subfolders")
    parser.add_argument('--use_d65', action='store_true', help="Whether to multiply by D65 spectrum")
    parser.add_argument('--output_folder', type=str, default='../Dataset/Output_images', help="Folder to store output images")
    args = parser.parse_args()

    cie_path = '../Dataset/CIE'
    cie_data = load_cie_data(cie_path)

    subfolders = [f.path for f in os.scandir(args.dataset_folder) if f.is_dir()]

    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        output_subfolder = os.path.join(args.output_folder, subfolder_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        for root, _, files in os.walk(subfolder):
            for file in files:
                if file.endswith('.npy'):
                    image_file = os.path.join(root, file)
                    process_image(image_file, cie_data, args.use_d65, output_subfolder)

if __name__ == "__main__":
    main()
