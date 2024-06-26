#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch

# mode = train or val, path = "/content/Cityscapes/Cityspaces/"
class CityScapes(Dataset):
    def __init__(self, mode, use_pseudo_label=False):
        super(CityScapes, self).__init__()
        self.path = "/content/Cityscapes/Cityspaces/"
        self.mode = mode
        self.use_pseudo_label= use_pseudo_label
        self.data, self.label, self.pseudo_label = self.data_loader()
        self.transform_data = transforms.Compose([ 
            transforms.ToTensor(),                 # Converts the image to a tensor
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.width = 1024
        self.height = 512
              
    def __getitem__(self, idx):
        
        image = self.pil_loader(self.data[idx],  'RGB')  
        tensor_image = self.transform_data(image)

        if self.use_pseudo_label:
            # Load pseudo label
            pseudo_label = self.pil_loader(self.pseudo_label[idx],  'L')
            tensor_pseudo_label=torch.from_numpy(np.array(pseudo_label))
            return tensor_image, tensor_pseudo_label

        label = self.pil_loader(self.label[idx],  'L')
        tensor_label =torch.from_numpy(np.array(label))
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
        pseudo_label=[]
        types = ["images/", "gtFine/", "pseudolbl/"]
        for t in types:
            # check if path exists
            if os.path.exists(self.path+t):
                # get files from directory
                for root, dirs, files in os.walk(self.path+t):
                    if self.mode in root or self.use_pseudo_label:
                        for file in files:
                            file_path = os.path.join(root, file) 
                            # get path in mode
                            relative_path = os.path.relpath(file_path, self.path)
                            if self.mode in root and t=="images/":
                                data.append(relative_path)
                            elif self.use_pseudo_label and t=="pseudolbl/":
                                pseudo_label.append(relative_path)
                            elif self.mode in root and file_path.split("gtFine_")[1] == "labelTrainIds.png":
                                label.append(relative_path)
   
        return sorted(data), sorted(label), sorted(pseudo_label)
