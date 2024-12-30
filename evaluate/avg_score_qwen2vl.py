import json
import os
from pathlib import Path
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import argparse

# 加载模型和处理器
def load_model_and_processor(device="cuda"):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map={"": device}
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return model, processor

# 输入
def prepare_inputs(processor, messages):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs

# 推理过程
def generate_output(model, inputs, device="cuda", max_tokens=128):
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs,  max_new_tokens=max_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return generated_ids_trimmed

# 后处理和结果解码
def decode_output(processor, generated_ids_trimmed):
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

# 处理目录中的所有图片并计算平均分数
def process_images_from_directory(image_directory, device="cuda"):
    # 加载模型和处理器
    model, processor = load_model_and_processor(device)

    # 获取图片路径列表
    image_paths = [str(Path(image_directory) / img_name) for img_name in os.listdir(image_directory)
                   if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))] 

    total_score = 0
    num_images = 0

    # 使用 tqdm 进度条
    for image_full_path in tqdm(image_paths, desc="Processing images", unit="image", total=len(image_paths)):
        # 检查图片是否存在
        if not os.path.exists(image_full_path):
            print(f"Image {image_full_path} not found.")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_full_path},
                    {"type": "text", "text": """Score the authenticity of this face (0-100) and return it in JSON format:{"score":int}"""}
                ],
            }
        ]

        # 处理输入
        inputs = prepare_inputs(processor, messages)
        generated_ids_trimmed = generate_output(model, inputs, device=device, max_tokens=128)
        output_text = decode_output(processor, generated_ids_trimmed)

        try:
            # 解析并获取分数
            score_dict = json.loads(output_text[0].strip())
            score = score_dict.get('score', None)
            
            if score is not None:
                total_score += score
                num_images += 1
                print(f"Processed {image_full_path}, score: {score}")
            else:
                print(f"Processed {image_full_path}, could not extract score.")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {image_full_path}. Output: {output_text[0]}. Error: {e}")

    # 计算平均分数
    if num_images > 0:
        average_score = total_score / num_images
        print(f"\nProcessed {num_images} images. Average score: {average_score:.2f}")
    else:
        print("No valid images processed.")

    print(f"Finished processing {num_images} images.")
    return average_score



def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Process images from a directory")

    # 添加参数
    parser.add_argument(
        "--image_directory", 
        type=str,
        default="/home/ddwgroup/san/jmm/ControlledFaceGeneration/evaluate/dataset/generate/unplash-50/InstantID",  
        help="图片所在的文件夹路径"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",  
        help="使用的设备 (例如 'cuda:0')"
    )

    args = parser.parse_args()
    
    process_images_from_directory(
        image_directory=args.image_directory, 
        device=args.device
    )

if __name__ == "__main__":
    main()