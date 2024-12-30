#!/bin/bash

# 检查是否提供了目录作为参数
if [ "$#" -ne 1 ]; then
    echo "用法: $0 <directory>"
    exit 1
fi

# 目标目录
input_directory="$1"

# 检查目录是否存在
if [ ! -d "$input_directory" ]; then
    echo "错误: 目录 $input_directory 不存在."
    exit 1
fi

# 遍历目录中的每个 avif 文件
for avif_file in "$input_directory"/*.avif; do
    # 检查是否有 avif 文件
    if [ ! -e "$avif_file" ]; then
        echo "没有找到 avif 文件."
        exit 1
    fi
    
    # 获取文件名，不带扩展名
    filename=$(basename -- "$avif_file")
    filename_no_ext="${filename%.*}"

    # 生成输出文件名
    jpg_file="$input_directory/$filename_no_ext.jpg"

    # 转换 avif 文件到 jpg
    ffmpeg -i "$avif_file" "$jpg_file"

    echo "转换: $avif_file 到 $jpg_file"
done

echo "完成转换！"