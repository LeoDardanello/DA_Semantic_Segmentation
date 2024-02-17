#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
from PIL import Image
import numpy as np
from my_utils import FourierDomainAdaptation, toimage

class FDA(Dataset):
  def __init__(self, data_source, data_target, beta=0.01):
    super(FDA, self).__init__()
    self.path = "/content/GTA5/"
    self.width = 1024
    self.height = 512
    
    self.label = [data_source.dataset.label[i] for i in data_source.indices]
    self.data_source = [data_source.dataset.data[i] for i in data_source.indices]
    self.data_target = data_target
    
    self.fda= FourierDomainAdaptation(beta)

    self.transform_data = transforms.Compose([ 
      transforms.ToTensor(),                 # Converte l'immagine in un tensore
      transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

  def pil_loader(self, p, mode):
    with open(p, 'rb') as f:
      img = Image.open(f)
      return img.convert(mode).resize((self.width, self.height), Image.NEAREST)
  

  def __getitem__(self, idx): 
    src = self.pil_loader(self.path+self.data_source[idx], 'RGB')
    file_path =f"FDA/{self.data_source[idx].split('/')[-1]}"

    if not os.path.exists("/content/GTA5/FDA/"):
        os.makedirs("/content/GTA5/FDA/")  
    if not os.path.exists(self.path+file_path): 
         
        trg= self.pil_loader("/content/Cityscapes/Cityspaces/"+self.data_target[idx], "RGB")
        img = self.fda(src, trg)
        image_fda = toimage(img, cmin=0.0, cmax=255.0)
        #saving FDA image creted with data_source[idx] and data_target[idx]
        image_fda.save(self.path+file_path)
    else:
        #if exist FDA imges created with data_source[idx] and data_target[idx], load it
        image_fda = self.pil_loader(self.path+file_path, 'RGB')

    label = self.pil_loader(self.path+self.label[idx], 'L')
    tensor_image = self.transform_data(src)
    tensor_image_fda = self.transform_data(image_fda)
    tensor_label = torch.from_numpy(np.array(label))  
    return tensor_image, tensor_label, tensor_image_fda

  def __len__(self):
    return min(len(self.data_source), len(self.data_target))
    