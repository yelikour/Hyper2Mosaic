# Hyper2Mosaic Tools

本文件夹包含用于处理超光谱图像数据的各种工具脚本。以下是每个文件的简要介绍和使用说明。

## 文件列表

- `check_individual_npy_sizes.py`
- `check_npy_files.py`
- `extract_channels.py`
- `Hyper2Mosaic.py`
- `Hyper2RGB.py`
- `Hyper2RGBAll.py`
- `notes.txt`
---
## 文件说明

### check_individual_npy_sizes.py
检查单个 `.npy` 文件的尺寸，以确保所有文件具有一致的形状。
#### 用法
```
python check_individual_npy_sizes.py --folder /path/to/npy_files
```

### check_npy_files.py
检查 `.npy` 文件的完整性和一致性。
#### 用法
```
python check_npy_files.py --folder /path/to/npy_files
```

### extract_channels.py
从超光谱图像中提取特定的通道。

#### 用法
```
python extract_channels.py --input /path/to/input_file.npy --channels 1 5 10 --output /path/to/output_file.npy
```
### Hyper2Mosaic.py
将超光谱图像转换为马赛克图像。
#### 用法
```
python Hyper2Mosaic.py --input /path/to/input_file.npy --output /path/to/output_file.png
```

### Hyper2RGB.py
将单个超光谱图像转换为RGB图像。
####用法
```
python Hyper2RGB.py --image_file /path/to/input_file.npy --use_d65 --output_folder /path/to/output_folder
```

### Hyper2RGBAll.py
遍历数据集中所有文件夹的所有文件，将所有超光谱图像转换为RGB图像。
#### 用法
```
python Hyper2RGBAll.py --dataset_folder ../Dataset/Extracted_Data --use_d65 --output_folder ../Dataset/Output_images
```
### notes.txt
记录一些使用工具时的笔记和注意事项。

## 备注
请根据实际需求调整脚本中的参数和路径。如果有任何问题或改进建议，请联系开发者。


