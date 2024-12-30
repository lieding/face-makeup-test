import os

def rename_files_in_directory(directory_path):
    try:
        # 获取目录下的所有文件
        files = sorted(os.listdir(directory_path))

        # 过滤出文件（不包括子目录）
        files = [f for f in files if os.path.isfile(os.path.join(directory_path, f))]

        # 遍历文件并重命名
        for i, filename in enumerate(files):
            # 生成新的四位数文件名
            new_name = f"{i:04}.jpg"
            
            # 构建完整路径
            old_file = os.path.join(directory_path, filename)
            new_file = os.path.join(directory_path, new_name)

            # 重命名文件
            os.rename(old_file, new_file)
            print(f"Renamed '{filename}' to '{new_name}'")

        print("All files have been successfully renamed.")
    except Exception as e:
        print(f"An error occurred: {e}")

# 示例调用
directory_path = "/home/ddwgroup/san/jmm/ControlledFaceGeneration/evaluate/dataset/unplash-o-t"
rename_files_in_directory(directory_path)