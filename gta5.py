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

def one_hot_it(label, label_info):
	semantic_map = np.zeros(label.shape[1:])
	for index, info in enumerate(label_info):
		color = info[-3:]
		class_map = np.all(label == color.reshape(3, 1, 1), axis=0)
		semantic_map[class_map] = index
	pprint(semantic_map)
	return torch.from_numpy(semantic_map)

def get_label_info(csv_path):
  # return label -> {label_name: [r_value, g_value, b_value, ...}
  ann = pd.read_csv(csv_path)
  label = {}
  for iter, row in ann.iterrows():
    label_name = row[0]
    label_id = row[1]
    rgb_color = row[2]
    label[label_name] = [label_id] +  list(rgb_color)
  return label

label_info = get_label_info('/content/DA_Semantic_Segmentation/GTA.csv') 

class Gta5(Dataset):
    def __init__(self, mode, args):
        super(Gta5, self).__init__()
        self.path = "/content/GTA5/"
        self.mode = mode
        self.data, self.label = self.data_loader()
        self.transform_data = transforms.Compose([ 
            # transforms.Resize((args.crop_height,args.crop_width)), # Ridimensiona l'immagine
            transforms.ToTensor(),                 # Converte l'immagine in un tensore
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.width=args.crop_width
        self.height = args.crop_height

    def __getitem__(self, idx):
        image = self.pil_loader(self.data[idx], 'RGB')
        label = self.pil_loader(self.label[idx], 'RGB')
        tensor_image = self.transform_data(image)
        tensor_label = one_hot_it(np.array(label), label_info)   
        return tensor_image, tensor_label 

    def __len__(self):
        return len(self.data)
    
    def pil_loader(self, p, mode):
        with open(self.path+p, 'rb') as f:
            img = Image.open(f)
            return img.convert(mode).resize((self.width, self.height), Image.NEAREST)

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


