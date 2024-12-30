from PIL import Image
import os

def convert_images_to_jpg(folder_path, output_folder):
    # 如果输出文件夹不存在，则创建该文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中所有文件
    for filename in os.listdir(folder_path):
        input_file = os.path.join(folder_path, filename)

        # 检查是否是文件而不是目录
        if os.path.isfile(input_file):
            # 分离文件名和扩展名
            name, ext = os.path.splitext(filename)
            ext = ext.lower()  # 将扩展名统一转换为小写

            # 只处理图片文件（可以根据需要扩展支持的格式）
            if ext in [".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".gif"]: 
                try:
                    # 打开图像并转换为 .jpg
                    with Image.open(input_file) as img:
                        img = img.convert("RGB")  # 转换为 RGB 模式，JPEG 不支持透明通道
                        output_file = os.path.join(output_folder, f"{name}.jpg")
                        img.save(output_file, "JPEG")
                        print(f"Converted: {input_file} -> {output_file}")
                except Exception as e:
                    print(f"Failed to convert {input_file}: {e}")

# 输入文件夹路径
input_folder_path = ""  # 修改为你的输入文件夹路径
output_folder_path = ""  # 修改为你的输出文件夹路径
convert_images_to_jpg(input_folder_path, output_folder_path)