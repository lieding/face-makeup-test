import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from tqdm import tqdm
from FaceMakeUp.facemakeup.facemakeup import FaceMakeUp
from FaceMakeUp.facemakeup.FaceMakeUp_Pipline import FaceMakeUp_Pipline
from insightface.app import FaceAnalysis
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.utils import face_align
import cv2
import numpy as np
import os
from diffusers.models import ControlNetModel
from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
import cv2
import torch
import numpy as np
from PIL import Image
import math
from insightface.app import FaceAnalysis
import json
import argparse
import traceback
from pathlib import Path

# 设置环境变量

# 初始化模型路径和参数
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
lora_rank = 128
base_model_path = "/home/ddwgroup/workplace/model/SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
vae = AutoencoderKL.from_pretrained(vae_model_path)
unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
controlnet = ControlNetModel.from_unet(unet)
pipe = FaceMakeUp_Pipline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float32,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
)
pipe.vae = vae
pipe.noise_scheduler = noise_scheduler
pipe.unet = unet
pipe.safety_checker = None
pipe.requires_safety_checker = False

prompts = [
 "a person wearing a white shirt",
 "a person with a red hat",
 "a person in a long coat",
 "a person wearing glasses",
 "a person with a leather bag",
 "a person standing in a park",
 "a person in a cozy room",
 "a person near a sunny beach",
 "a person standing under streetlights",
 "a person in a busy market",
 "a person running on a track",
 "a person holding a cup of coffee",
 "a person reading a book",
 "a person playing a guitar",
 "a person talking on the phone",
 "a person dressed in formal attire",
 "a person wearing sportswear",
 "a person traditional cultural clothing",
 "a person in casual jeans and a t-shirt",
 "a person dressed for a wedding",
]

# 绘制关键点函数
def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)
    
    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])
    
    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]
        
        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)
    
    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)
    
    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

def get_file_anme(item, crop):

    return item['face'][0] if crop else item['image'] 


# 初始化人脸分析
def init_face_analysis(app_name, providers, ctx_id, det_size=(512, 512)):
    app = FaceAnalysis(name=app_name, providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Face Generation Script")
    parser.add_argument('-p1', '--scale', type=float, default=0.6, help='第一个参数')
    parser.add_argument('-p2', '--s_scale', type=float, default=1.0, help='第二个参数')
    parser.add_argument('-p3', '--controlnet_conditioning_scale', type=float, default=0.2, help='第三个参数')
    parser.add_argument('--input_folder', type=str, default="/home/ddwgroup/san/xh/Facecaption_datasets/test_1k/images/img_orial/", help='输入文件夹路径')
    parser.add_argument('--output_folder', type=str, default="/home/ddwgroup/san/jmm/ControlledFaceGeneration/evaluate/dataset/JFIDEVALUATE/", help='输出文件夹路径')
    parser.add_argument('--extractor', type=str, default="buffalo_l", choices=['antelopev2', 'buffalo_l'], help='模型选择')
    parser.add_argument('--crop', action='store_true', help='是否裁剪')
    parser.add_argument('--use_folder', action='store_true', help='是否直接从文件夹读取图片')
    parser.add_argument('--device', type=str, default="cuda:0",help='gpu选择')
    parser.add_argument('--resize', action='store_true',help='做resize')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.use_folder:
        # 从文件夹中读取图片文件名
        image_files = [f for f in os.listdir(args.input_folder) if os.path.isfile(os.path.join(args.input_folder, f))]
        data = [{'image': img_file, 'face': [img_file],'image_short_caption': ['']} for img_file in image_files]  # 使用空字符串作为prompt
    else:
        # 从JSON文件中读取数据
        with open('/home/ddwgroup/san/xh/Facecaption_datasets/test_1k/test_1k.json', 'r') as file:
            data = json.load(file)[:10]
    
    folder_path = args.input_folder
    output_folder_base = args.output_folder
    extractor = args.extractor
    crop = args.crop
    use_folder = args.use_folder
    device = args.device
    resize = args.resize
    # 模型文件路径
    model_path = "/home/ddwgroup/san/jmm/ControlledFaceGeneration/ipAdaptet/JFID_v4/checkpoint-530000/model.bin"

    model = FaceMakeUp(pipe, image_encoder_path, model_path, device, lora_rank = lora_rank, torch_dtype=torch.float32)



    faceid_embeds_list = {}
    face_kps_list = {}
    faces_list = {}

    # 初始化人脸分析
    app = init_face_analysis(extractor, ['CUDAExecutionProvider', 'CPUExecutionProvider'], 0)

    # 创建一个列表，记录未成功处理的项
    failed_items = []
    # ------------------------------------------------------------------------
    for item in tqdm(data, desc="Extracting face embeddings"):
        filename = get_file_anme(item, crop)
        file_path = os.path.join(folder_path, filename)
        
        # 读取 img 并转换为 BGR 格式
        img = cv2.imread(file_path)
        if img is None:
            # 如果文件未找到或无法读取，记录失败项
            failed_items.append(item)
            continue
        
        img_COLOR_BGR2RGB = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
        
        try:
            # 提取 faces
            faces = app.get(img_COLOR_BGR2RGB)
        except Exception as e:
            # 如果提取人脸时出错，记录失败项
            failed_items.append(item)
            continue

        if len(faces) == 0:
            # 如果未检测到人脸，记录失败项
            failed_items.append(item)
            continue

        # 提取 faceid_embeds
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        faceid_embeds_list[filename] = faceid_embeds

    # 在循环结束后，删除失败项
    data = [item for item in data if item not in failed_items]
    # ------------------------------------------------------------------------
    app = init_face_analysis('antelopev2', ['CUDAExecutionProvider', 'CPUExecutionProvider'], 0)
    # 提取 face_kps_list
    for item in tqdm(data, desc="Extracting face keypoints"):
        filename = get_file_anme(item,crop)
        file_path = os.path.join(folder_path, filename)
        prompt = item['image_short_caption'][0]  # 使用短标题作为prompt

        # 读取 img 并转换为 RGB 格式
        img_RGB = Image.open(file_path).convert("RGB")

        # 提取 faces
        img_COLOR_BGR2RGB = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
        faces = app.get(img_COLOR_BGR2RGB)
        if len(faces) == 0:
            failed_items.append(item)
            continue    

        # 提取 face_kps_list
        face_kps = draw_kps(img_RGB, faces[0]['kps'])
        face_kps = face_kps.resize([512, 512])
        face_kps_list[filename] = face_kps

    # 过滤data
    data = [item for item in data if item not in failed_items]
    # ------------------------------------------------------------------------
    app = init_face_analysis("buffalo_l", ['CUDAExecutionProvider', 'CPUExecutionProvider'], 0)
    def rtn_face_get(self, img, face):
        aimg = face_align.norm_crop(img, landmark=face.kps, image_size=512)
        face.crop_face = aimg
        return face
    
    ArcFaceONNX.get = rtn_face_get
    
    for item in tqdm(data,desc="Extracting faces_list"):
        filename = get_file_anme(item,crop)
        prompt = item['image_short_caption'][0]
        file_path = os.path.join(folder_path, filename)

        img = cv2.imread(file_path)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
        ls = app.get(img)
        if (len(ls)==0):
            failed_items.append(item)
            continue
        # faces = ls[0].crop_face if crop else Image.open(file_path).convert("RGB")
        faces = Image.open(file_path).convert("RGB")
        faces_list[filename] = faces if not resize else faces.resize((512, 512))

    # 过滤data
    data = [item for item in data if item not in failed_items]

    # 使用参数并进行生成
    scale = args.scale
    s_scale = args.s_scale
    controlnet_conditioning_scale = args.controlnet_conditioning_scale
    crop_s = "crop" if crop else "x"
    resize_s = "resize" if resize else "x"
    data_type = 50 if use_folder else 1000
    # saveFloder = f"{output_folder_base}/JFID_V4_face_1.0_{data_type}_{scale}_{s_scale}_{controlnet_conditioning_scale}_{extractor}_{crop_s}_{resize_s}/"
    saveFloder = output_folder_base
    os.makedirs(saveFloder, exist_ok=True)
    
    # 遍历数据并应用新的 prompts
    for item in data:
       for idx, prompt_text in enumerate(prompts):
            try:
                filename = get_file_anme(item, crop)
                prompt = prompt_text  # 使用新定义的 prompt
                face_kps = face_kps_list[filename]
                faces = faces_list[filename]
                faceid_embeds = faceid_embeds_list[filename]
                
                # 调用模型生成图像
                images = model.generate(
                    prompt="best quality"+prompt,  # 使用每次的新 prompt
                    negative_prompt="black image, Easy Negative,worst quality,low quality, lowers,monochrome,grayscales,skin spots,acnes,skin blemishes,age spot,6 more fingers on one hand,deformity,bad legs,error legs,bad feet,malformed limbs,extra limbs,ugly,poorly drawn hands,poorly drawn feet.poorly drawn face,text,mutilated,extra fingers,mutated hands,mutation,bad anatomy,cloned face,disfigured,fused fingers",
                    image=face_kps,             # antelopev2
                    face_image=faces,           # buffalo_l
                    faceid_embeds=faceid_embeds,# antelopev2
                    shortcut=True,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    scale=scale,
                    s_scale=s_scale,
                    seed=42,
                    num_samples=1,
                    width=512,
                    height=512,
                    num_inference_steps=50,
                )


                # 获取文件名和扩展名
                name, ext = os.path.splitext(filename)

                # 构造带索引的文件名，序号后缀从 001 开始
                numbered_filename = f"{name}_{idx + 1:03d}{ext}"

                # 创建保存路径的文件夹
                os.makedirs(Path(os.path.join(saveFloder, numbered_filename)).parent, exist_ok=True)
                # 保存生成的图像
                images[0].save(os.path.join(saveFloder, numbered_filename))
            except Exception as e:
                print(f"Error processing {filename} with prompt '{prompt_text}': {e}")
                traceback.print_exc()

if __name__ == "__main__":
    main()
