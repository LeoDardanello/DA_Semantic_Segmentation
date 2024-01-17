#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
from PIL import Image
import numpy as np
import pandas as pd
from utils import FourierDomainAdaptation
from tqdm.auto import tqdm

class FDA(Dataset):
  def __init__(self, data_source, data_target, label_source, beta=0.01):
    super(FDA, self).__init__()
    self.path = "/content/GTA5/"
    self.width = 1024
    self.height = 512
    self.data = self.to_FDA(data_source, data_target, beta)
    self.label = label_source
    self.transform_data = transforms.Compose([ 
      transforms.ToTensor(),                 # Converte l'immagine in un tensore
      transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

  def pil_loader(self, p, mode):
    with open(p, 'rb') as f:
      img = Image.open(f)
      return img.convert(mode).resize((self.width, self.height), Image.NEAREST)
  
  def to_FDA(self, data_source, data_target, beta):
    if not os.path.exists("/content/GTA5/FDA/"):
      os.makedirs("/content/GTA5/FDA/")
    print("Generating FDA images..")
    img_list = []
    fda = FourierDomainAdaptation(beta)
    for i, (src, trg) in tqdm(enumerate(zip(data_source, data_target))):
      file_path =f"/content/GTA5/FDA/{str(i+1).zfill(5)}.png"
      img_list.append(f"FDA/{str(i+1).zfill(5)}.png")
      if not os.path.exists(file_path):
        src=self.pil_loader("/content/GTA5/"+src, "RGB")
        trg= self.pil_loader("/content/Cityscapes/Cityspaces/"+trg, "RGB")
        img = fda(src, trg)
        conv_img = Image.fromarray(np.uint8(img))
        conv_img.save(file_path)
    return img_list

  def __getitem__(self, idx):
    image = self.pil_loader(self.path+self.data[idx], 'RGB')
    label = self.pil_loader(self.path+self.label[idx], 'L')
    tensor_image = self.transform_data(image)
    tensor_label = torch.from_numpy(np.array(label))  
    return tensor_image, tensor_label 

  def __len__(self):
    return len(self.data)
    
    


