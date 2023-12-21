#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

def pil_loader(p, mode):
    with open(p, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)

# mode = train or val, path = "/content/Cityscapes/Cityspaces/"
class CityScapes(Dataset):
    def __init__(self, mode,args):
        super(CityScapes, self).__init__()
        self.path = "/content/Cityscapes/Cityspaces/"
        self.mode = mode
        self.data, self.label = self.data_loader()
        self.preprocess = transforms.Compose([
            transforms.Resize((args.crop_height,args.crop_width)), # Ridimensiona l'immagine
            transforms.ToTensor(),                 # Converte l'immagine in un tensore

        ])
        
        # ImageNet pretraining statistics
        self.normalizer=transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
    def __getitem__(self, idx):
        image = pil_loader(self.path+self.data[idx], 'RGB')
        
        label = pil_loader(self.path+self.label[idx], 'L')
        tensor_image = self.preprocess(image)
        
        return self.normalizer(tensor_image), self.preprocess(label)

    def __len__(self):
        return len(self.data)

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
