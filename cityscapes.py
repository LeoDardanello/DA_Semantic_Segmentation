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
    def __init__(self, mode,args):
        super(CityScapes, self).__init__()
        self.path = "/content/Cityscapes/Cityspaces/"
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
        image = self.pil_loader(self.data[idx],  'RGB')
        label = self.pil_loader(self.label[idx],  'L')

        tensor_image = self.transform_data(image)
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
        types = ["images/", "gtFine/"]
        for t in types:
            # check if path exists
            if os.path.exists(self.path+t):
                # get files from directory
                for root, dirs, files in os.walk(self.path+t):
                    if self.mode in root:
                        for file in files:
                            file_path = os.path.join(root, file)
                            # get path in mode
                            relative_path = os.path.relpath(file_path, self.path)
                            if t=="images/":
                                data.append(relative_path)
                            else:
                                if file_path.split("gtFine_")[1] == "labelTrainIds.png":
                                    label.append(relative_path)
        return sorted(data), sorted(label)
