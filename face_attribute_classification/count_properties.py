import os
import argparse
import csv
from PIL import Image
from glob import glob
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from utils.utils import torch_init_model
from model import AttClsModel

def load_atts(path):
    with open(path, 'r') as f:
        for line in f:
            atts = line.split()
    return atts

def append_list_to_csv(file_path, data_list):
    with open(file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data_list)

def center_crop(image):
    width, height = image.size
    
    if width > height:
        crop_size = height
        start_x = (width - height) // 2
        start_y = 0
    else:
        crop_size = width
        start_x = 0
        start_y = (height - width) // 2
    
    cropped_image = image.crop((start_x, start_y, start_x + crop_size, start_y + crop_size))
    return cropped_image

def process_images(photo_path_list, args):
    
    # 使用单个GPU或CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = AttClsModel(args, device=device)
    model.to(device)
    torch_init_model(model, os.path.join(args.checkpoint_dir, 'best_model.pth'))

    # 拼接图像转换流程
    transform_list = [
        transforms.Resize((args.img_size, args.img_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    img2tensor = transforms.Compose(transform_list)

    # 加载属性映射
    att_map = load_atts(args.att_path)
    att_map.insert(0, 'id')
    # 如果需要，可以将属性表头写入 CSV
    # append_list_to_csv(args.csv_path, [att_map])
    
    avg = 0

    # 循环处理所有图片
    total_iterations = len(photo_path_list)
    
    for i, photo_path in tqdm(enumerate(photo_path_list), total=total_iterations, desc="Processing", unit="image"):
        img = Image.open(photo_path)
        img = center_crop(img)
        img = img2tensor(img)

        # 推断模型
        model.eval()
        att_preds = [os.path.basename(photo_path)]  # 保存图片名
        with torch.no_grad():
            logits = model(img.to(device).unsqueeze(0))  # 添加 batch 维度
            sigmoid_probs = torch.sigmoid(logits)[0]
            sigmoid_probs = sigmoid_probs.detach().cpu().numpy()

        # 对每个属性设置阈值
        for j in range(len(att_map) - 1):
            if float(sigmoid_probs[j]) > 0.85:
                att_preds.append(1)
            else:
                att_preds.append(-1)
        
        # # 如果满足条件，保存结果到CSV
        # if att_preds.count(1) >= 5:
        #     append_list_to_csv(csv_path, [att_preds])
        # print(att_preds.count(1))
        avg = avg + att_preds.count(1) / total_iterations
    return avg

# def main():
#     # 获取脚本所在目录路径
#     script_dir = os.path.dirname(os.path.realpath(__file__))

#     # 定义命令行参数
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_type', type=str, default='resnet50')
#     parser.add_argument('--img_size', type=int, default=256)
#     parser.add_argument('--float16', type=bool, default=False)

#     # 目录和文件路径通过拼接脚本路径来设置为相对路径
#     parser.add_argument('--img_path', type=str, 
#                         default='/home/guest/workplace/zyx/FakeFaceCaption15M/imgs/test_space')  
#     parser.add_argument('--att_path', type=str,
#                         default=os.path.join(script_dir, 'data_list/att_map.txt'))  # 相对路径
#     parser.add_argument('--checkpoint_dir', type=str,
#                         default=os.path.join(script_dir, 'FAC_resnet50_AW_V1'))  # 相对路径

#     args = parser.parse_args()
# cou 
#     # 加载图片路径
#     photo_paths = sorted(glob(os.path.join(args.img_path, '*')))
    
#     # 处理所有图片
#     res =  process_images(photo_paths, args)
#     print(f"average properties: {res:.2f}")
#     print('All images processed.')


def main():
    # 获取脚本所在目录路径
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='resnet50')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--float16', type=bool, default=False)

    # 目录和文件路径通过拼接脚本路径来设置为相对路径
    parser.add_argument('--img_path', type=str, 
                        default='/home/guest/workplace/zyx/FakeFaceCaption15M/imgs/test_space')  
    parser.add_argument('--att_path', type=str,
                        default=os.path.join(script_dir, 'data_list/att_map.txt'))  # 相对路径
    parser.add_argument('--checkpoint_dir', type=str,
                        default=os.path.join(script_dir, 'FAC_resnet50_AW_V1'))  # 相对路径

    args = parser.parse_args()

    # 加载图片路径
    # photo_paths = sorted(glob(os.path.join(args.img_path, '*')))
    photo_paths = sorted(glob(os.path.join(args.img_path, '*.jpg')))
        
    # 处理所有图片
    res = process_images(photo_paths, args)
    print(f"average properties: {res:.2f}")
    print('All images processed.')
if __name__ == '__main__':
    main()