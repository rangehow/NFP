#!/bin/bash

# 设置初始的 CUDA_VISIBLE_DEVICES 值
cuda_device=0

# 获取指定路径下所有文件夹名字
folder_path="/data/ruanjh/best_training_method/T5_V3_4gram_again_celoss"
folders=$(find "$(realpath "$folder_path")" -mindepth 1 -maxdepth 1 -type d)
echo $folders
# 传递每个文件夹名字作为参数给 Python 文件，并指定 CUDA_VISIBLE_DEVICES
for folder in $folders; do
    echo $folder
    CUDA_VISIBLE_DEVICES=$cuda_device python your_python_script.py "$folder"
    ((cuda_device++))  # 递增 CUDA_VISIBLE_DEVICES 值
done
