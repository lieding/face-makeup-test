


from transformers import ViTImageProcessor, ViTModel, CLIPVisionModel, CLIPImageProcessor
from PIL import Image
import json
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm   
from insightface.app import FaceAnalysis
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
class Metric:
    def __init__(self, mode, ann_path ,real_path, fake_path, sample_size = 1000):
        
        if mode == 'CLIP-I':
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").cuda()
            self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        if mode == 'DINO':
            self.processor = ViTImageProcessor.from_pretrained("facebook/dino-vits16")
            self.model = ViTModel.from_pretrained("facebook/dino-vits16").cuda()
        if mode == 'FaceSim':
            self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))

        
        self.device = 'cuda'
        # self.ann_path = ann_path
        # anns = json.load(open(ann_path, 'r'))[:sample_size]

        if mode == 'CLIP-I' or mode == 'DINO':
          
            self.fake_imgs = []
            self.real_imgs = []
            for filename  in  os.listdir(real_path):
                fake_img = Image.open(os.path.join(fake_path,filename)).convert('RGB')
                real_img = Image.open(os.path.join(real_path,filename)).convert('RGB')
                fake_img = self.processor(images=fake_img, return_tensors="pt")['pixel_values']
                real_img = self.processor(images=real_img, return_tensors="pt")['pixel_values']
                self.fake_imgs.append(fake_img)
                self.real_imgs.append(real_img)
            
        else:
            self.fake_imgs = []
            self.real_imgs = []
            for filename  in  os.listdir(real_path):
                try:
                    fake_img =cv2.imread(os.path.join(fake_path,filename))
                    real_img = cv2.imread(os.path.join(real_path,filename))
                    fake_img = cv2.cvtColor(np.asarray(fake_img), cv2.COLOR_BGR2RGB)
                    real_img = cv2.cvtColor(np.asarray(real_img), cv2.COLOR_BGR2RGB)
                
                    fake_img_faces = torch.from_numpy(self.app.get(fake_img)[0].normed_embedding).unsqueeze(0)
                    real_img_faces = torch.from_numpy(self.app.get(real_img)[0].normed_embedding).unsqueeze(0)
                    self.fake_imgs.append(fake_img_faces)
                    self.real_imgs.append(real_img_faces)
                except:
                    pass


    
    def evaluateDINO(self):
        """
            DINO评估模型生成的图像质量。
            Args:
                无
            Returns:
                float: 生成图像与真实图像之间嵌入向量的平均余弦相似度。
        """
        
        fake_embedding = self.model(pixel_values=torch.cat(self.fake_imgs).to(self.device), output_hidden_states=True).pooler_output
        real_embedding = self.model(pixel_values=torch.cat(self.real_imgs).to(self.device), output_hidden_states=True).pooler_output
        sim = torch.cosine_similarity(fake_embedding, real_embedding)
        return sim.mean().item()
    
    def evaluateCLIP_I(self):
        """
            CLIP-I评估模型的性能。
            Args:
                无
            Returns:
                float: 计算得到的平均余弦相似度，用于评估模型生成的假图像与真实图像之间的相似度。
        """

        fake_embedding = self.model(pixel_values=torch.cat(self.fake_imgs).to(self.device), output_hidden_states=True).pooler_output
        
        real_embedding = self.model(pixel_values=torch.cat(self.real_imgs).to(self.device), output_hidden_states=True).pooler_output
        sim = torch.cosine_similarity(fake_embedding, real_embedding)
        return sim.mean().item()
    
    def evaluateFaceSim(self):
        
        sim = []
        for fake_emb , real_emb in zip(self.fake_imgs, self.real_imgs):
            sim.append(self.app.models['recognition'].compute_sim(fake_emb, real_emb).item())
        return sum(sim)/len(sim) if sim else 0
        
        # 两种方式都可以
        # fake_embedding = torch.cat(self.fake_imgs)
        # real_embedding = torch.cat(self.real_imgs)
        # sim = torch.cosine_similarity(fake_embedding, real_embedding)
        # return sim.mean().item()
if __name__ == '__main__':
    # FaceSim DINO CLIP-I
    mode = 'FaceSim' 
    ann_path = '/home/ubuntu/san/xh/caption20w/HQ_test20w.json'  
    real_path = '/home/ddwgroup/san/jmm/ControlledFaceGeneration/evaluate/dataset/unplash-50-resize'
    fake_path = '/home/ddwgroup/san/jmm/ControlledFaceGeneration/evaluate/dataset/generate/unplash-50/jfid_controlnet_0_1.0_1.0_0.3'
    #fake_path = '/home/ddwgroup/san/jmm/ControlledFaceGeneration/evaluate/dataset/generate/unplash-50/jfid_controlnet_0_1.0_1.0_0.3'
    # fake_path = '/home/ubuntu/san/jmm/ControlledFaceGeneration/evaluate/dataset/generate/unplash-50/ipAdapterFaceIdPlus-388000'
    
    metric =  Metric(mode, ann_path, real_path, fake_path)
    print(metric.evaluateFaceSim())

    mode = 'DINO' 
    metric =  Metric(mode, ann_path, real_path, fake_path)
    print(metric.evaluateDINO())

    mode = 'CLIP-I' 
    metric =  Metric(mode, ann_path, real_path, fake_path)
    print(metric.evaluateCLIP_I()) 
