#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
from collections import namedtuple
from pprint import pprint
from PIL import Image
import numpy as np
import pandas as pd
from utils import from_RGB_to_LabelID

class GTA5(Dataset):
    def __init__(self, mode):
        super(GTA5, self).__init__()
        self.path = "/content/GTA5/"
        self.mode = mode
        self.data, self.label_colored = self.data_loader()
        self.transform_data = transforms.Compose([ 
            transforms.ToTensor(),                 # Converte l'immagine in un tensore
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.width = 1024
        self.height = 512
        self.label=from_RGB_to_LabelID(self.label_colored,self.path,self.height,self.width)

    def pil_loader(self, p, mode):
        with open(self.path+p, 'rb') as f:
            img = Image.open(f)
            return img.convert(mode).resize((self.width, self.height), Image.NEAREST)

    def __getitem__(self, idx):
        image = self.pil_loader(self.data[idx], 'RGB')
        label = self.pil_loader(self.label[idx], 'L')
        tensor_image = self.transform_data(image)
        tensor_label = torch.from_numpy(np.array(label))  
        return tensor_image, tensor_label 

    def __len__(self):
        return len(self.data)
    
    def __add__(self, path):
        self.data.append(path[0])
        self.label.append(path[1])
    
    def data_loader(self):
        data= []
        label = []
        types = ["images/", "labels/"]
        
        for t in types:
            for root, dirs, files in os.walk(self.path+t):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.path)
                    if t=="images/":
                        data.append(relative_path)
                    else:
                        label.append(relative_path)
        return sorted(data), sorted(label)


