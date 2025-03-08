import gradio as gr
import os
from diffusers import  DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from facemakeup.facemakeup import FaceMakeUp
from facemakeup.FaceMakeUp_Pipline import FaceMakeUp_Pipline
from PIL import Image
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.utils import face_align
from insightface.app import FaceAnalysis
from diffusers.models import ControlNetModel
from urllib import parse
import re
import requests
import html
import math
import numpy as np
from PIL import Image
import random
import cv2
import torch


# 初始化人脸分析
def init_face_analysis(app_name, providers, ctx_id, det_size=(512, 512)):
    app = FaceAnalysis(name=app_name, providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app

# model,tokenizer, device,config = init()
GOOGLE_TRANSLATE_URL = 'http://translate.google.com/m?q=%s&tl=%s&sl=%s'

def translate(text, to_language="auto", text_language="auto"):

    text = parse.quote(text)
    url = GOOGLE_TRANSLATE_URL % (text, to_language, text_language)
    response = requests.get(url)
    data = response.text
    expr = r'(?s)class="(?:t0|result-container)">(.*?)<'
    result = re.findall(expr, data)
    if (len(result) == 0):
        return ""
    return html.unescape(result[0])

def greet(person_txt,pro_image, scale_ip , scale_control, guidance_scale):
    
    # print(trans)
    # print("before:"+person_txt)
    # person_txt = translate(person_txt,  "en","zh-CN")  if trans else person_txt
    # print("trans:"+person_txt)

    
    images =  evaluate(person_txt,  pipeline=model, pro_image=pro_image, scale_ip = float(scale_ip), scale_control = float(scale_control), guidance_scale = float(guidance_scale))

    return images



@torch.inference_mode()
def evaluate(text,  pipeline=None, pro_image=None, scale_ip = 1, scale_control = 1, guidance_scale = 7.5):
    
  # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    

    
    negative_prompt = "black image, Easy Negative,worst quality,low quality, lowers,monochrome,grayscales,skin spots,acnes,skin blemishes,age spot,6 more fingers on one hand,deformity,bad legs,error legs,bad feet,malformed limbs,extra limbs,ugly,poorly drawn hands,poorly drawn feet.poorly drawn face,text,mutilated,extra fingers,mutated hands,mutation,bad anatomy,cloned face,disfigured,fused fingers"

    # negative_prompt = ""


    if pro_image is not None:
        pro_image = Image.fromarray(pro_image) # 将 numpy 数组转为图片
        pro_image.save("tmp.png")
        pro_image = pro_image.resize((512, 512))  # 将图片调整到 256x256
    

    app = init_face_analysis('antelopev2', ['CUDAExecutionProvider', 'CPUExecutionProvider'], 0)

     # 提取 faces
    img_COLOR_BGR2RGB = cv2.cvtColor(np.asarray(pro_image), cv2.COLOR_BGR2RGB)
    faces = app.get(img_COLOR_BGR2RGB)

    # 提取 face_kps_list
    face_kps = draw_kps(pro_image, faces[0]['kps'])
    face_kps.resize([512, 512])



    app = init_face_analysis("buffalo_l", ['CUDAExecutionProvider', 'CPUExecutionProvider'], 0)

    faces = app.get(img_COLOR_BGR2RGB)
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

    img =  cv2.imread("tmp.png")
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
    def rtn_face_get(self, img, face):
        aimg = face_align.norm_crop(img, landmark=face.kps, image_size=512)
        face.crop_face = aimg
        return face
    ArcFaceONNX.get = rtn_face_get
    
    ls = app.get(img)
    face = ls[0].crop_face

    def get(self, img, face):
        aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        face.embedding = self.get_feat(aimg).flatten()
        return face.embedding
    ArcFaceONNX.get = get
    # 调用模型生成图像
    seed = random.randint(1, 10000000)

    print(seed)
    global index___index
        
    seed = list_seeds[index___index]
    text = list_prompts[index___index]
    scale_control = list_scale_controls[index___index]
    scale_ip = list_scale_ips[index___index]
    gui = list_gras[index___index]
    negative_prompt = "black image, Easy Negative,worst quality,low quality, lowers,monochrome,grayscales,skin spots,acnes,skin blemishes,age spot,6 more fingers on one hand,deformity,bad legs,error legs,bad feet,malformed limbs,extra limbs,ugly,poorly drawn hands,poorly drawn feet.poorly drawn face,text,mutilated,extra fingers,mutated hands,mutation,bad anatomy,cloned face,disfigured,fused fingers" if index___index < 20 else ""

    index___index = index___index + 1

    images = model.generate(
                    prompt= text,  # 使用每次的新 prompt
                    negative_prompt=negative_prompt,
                    image=face_kps,             # antelopev2
                    face_image=face,           # buffalo_l
                    faceid_embeds=faceid_embeds,# antelopev2
                    shortcut=True,
                    controlnet_conditioning_scale=scale_control,
                    scale=scale_ip,
                    s_scale=1,
                    seed=seed,
                    num_samples=1,
                    guidance_scale=gui,
                    width=512,
                    height=512,
                    num_inference_steps=50,
                )
    
    
   
    
    return images[0]



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



if __name__== "__main__":
    
    base_model_path =  "stabilityai/stable-diffusion-xl-base-1.0"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    # ip_ckpt = "/home/guest/workplace/jmm/ControlFace/IP-Adapter/outputs/sd1.5-ip-adapter-face-id/checkpoint-156000/model.bin"
    # ip_ckpt = "/home/ubuntu/san/jmm/ControlledFaceGeneration/ipAdaptet/sd-ip_adapter-facecaption-faceid-sdxl/checkpoint-256000/model.bin"
    
    device = "cuda:7"
    
    list_prompts = [
                "a person holding a cup of coffee", 
                "a person  holding a cup of coffee near a sunny beach",
                "a person playing a guitar",
                "a person with glasses playing a guitar",
                "a person typing on a laptop",
                "a person with headphones typing on a laptop",
                "a person waving",
                "a person waving and smiling",
                "A man holding a cat",
                "A man With white shirt holding a cat",
                
                "a man with blue eyes",
                "a man with blue eyes holding a basketball",
                "An armored woman",
                "An armored woman with elf ears",
                "An astronaut in an asteroid field",
                "An astronaut with glasses in an asteroid field",
                "A priestess invokes protective spells in a sacred grove",
                "A priestess with hat invokes protective spells in a sacred grove",
                "A man in a golden field",
                "A man laughing in a golden field",
                

                "a Ukiyo-e painting of a person",
                "a paiting ofperson i VanGogh style",
                "a sketch of woman in black and white",
                "a paiting ofperson i VanGogh style",
                "A person in pixel art style",
                "A person in Baroque style",

                ]
    
    list_seeds = [ 
                7894886, 4975259,
                5699789, 6538638, 
                9154060, 3364981, 
                5112967, 4610930,
                5388684, 3188073,

                5258883,4634430,
                5699022,7391058,
                7831065,4535584,
                888040, 5147312,
                9291011,166948,


                1551350, 9965081,
                42428, 4672938,
                3034001,5124504,
                ]
    list_scale_controls = [
        
                0.6,0.6,
                0.4,0.4,
                0.2,0.2,
                0.2,0.2,
                0.6,0.6,
                
                0.6,0.6,
                0.6,0.6,
                0.2,0.2,
                0.2,0.2,
                0.2,0.2,

                0.1,0.1,
                0.1,0.1,
                0.1,0.1 
                ]
    list_scale_ips = [
                0.6,0.6,
                0.6,0.6,
                0.6,0.6,
                0.6,0.6,
                0.6,0.6,

                0.6,0.6,
                0.6,0.6,
                0.6,0.6,
                0.6,0.6,
                0.6,0.6,
  
                0.6,0.5,
                0.5, 0.5,
                0.1,0.1 
                ]
    
 
    list_gras = [ 
        7.5,7.5,
        7.5,7.5,
        7.5,7.5,
        7.5,7.5,
        7.5,7.5,
        
        7.5,7.5,
        7.5,7.5,
        7.5,7.5,
        7.5,7.5,
        7.5,7.5,
        
        7.5,7.5,
        7.5,7.5,
        10,10
    ]
        
    
    
    index___index = 10

    # 初始化模型路径和参数
    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    lora_rank = 128
    base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
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
    # 模型文件路径
    model_path = "./model.bin"

    model = FaceMakeUp(pipe, image_encoder_path, model_path, device, lora_rank = lora_rank, torch_dtype=torch.float32)


    # 启动Gradio界面
    # 启动 Gradio 界面
    css = """
        .my-custom-textbox input {  # 修改 Textbox 的字体大小
            font-size: 25px;
        }
        span.md.svelte-r3x3aw h1{
            font-size: xxx-large;
        }
        span.md.svelte-r3x3aw p{
            font-size: xx-large;
        }
        span.ml-2.svelte-3pzdsv{
            font-size: xx-large;
        }
        span.svelte-1gfkn6j{
            font-size: xx-large;
        }
        # textarea.scroll-hide.svelte-1f354aw{
        #     font-size: xx-large;
        # }
        label.svelte-1f354aw.container textarea{
            font-size: xx-large;
        }
    """
    demo = gr.Interface(
        fn=greet,
        inputs=[
            # gr.Checkbox(label="使用中文", value=True),
            gr.Textbox(lines=1, placeholder="input...", label="Prompt", elem_classes="my-custom-textbox"),
            gr.Image(label="ID image(main)"),
            gr.Slider(minimum = 0,maximum = 1, step=0.1 , value=0.6, label="ID Scale", info="Adjust the ID scale between 0 and 1"),
            gr.Slider(minimum = 0,maximum = 1, step=0.1 , value=0.6, label="Control Scale"),
            gr.Slider(0,10,value=7.5,label="Guidance Scale")
        ],
        outputs=gr.Image(label="Output"),
        css=css,
        title="FaceMaker-V0 @ddw2AIGROUP-CQUPT",
        description="FaceMaker-V0 is a tuning-free ID customization approach. FaceMaker maintains high ID fidelity.",
        examples=[
            ["a portrait of an man with glasses", "/home/ddwgroup/san/jmm/ControlledFaceGeneration/IP-Adapter/PuLID/1.png" ,0.6 ,0.6, 7.5],
            
        ]
    )
    demo.launch(server_name="0.0.0.0",share=True,server_port=8088)

