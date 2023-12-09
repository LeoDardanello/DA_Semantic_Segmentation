#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
import os
from PIL import Image

# mode = train or val, path = "/content/Cityscapes/Cityspaces/"  
class CityScapes(Dataset):
    def __init__(self, mode, path):
        super(CityScapes, self).__init__()
        self.path = path
        self.mode = mode
        self.data, self.label = self.data_loader(mode, path)


    def __getitem__(self, idx):
        image = pil_loader(self.path+"/"+self.data[idx])
        label = pil_loader(self.path+"/"+self.label[idx])
        return image, label

    def __len__(self):
        return len(self.data)
    
    def pil_loader(self):
        with open(self.path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    def data_loader(self, mode, folder_path):
        data= []
        label = []
        types = ["images/", "gtFine/"]
        for t in types:
            # Verifica se il percorso della cartella esiste
            if os.path.exists(folder_path+t):
                # Ottieni tutti i file nella cartella
                for root, dirs, files in os.walk(folder_path+t):
                    # Verifica se la cartella di destinazione è presente nella gerarchia del percorso
                    if mode in root:
                        for file in files:
                            # Crea il percorso completo del file
                            file_path = os.path.join(root, file)

                            # Ottieni il percorso relativo alla cartella "train"
                            relative_path = os.path.relpath(file_path, folder_path)
                            if t=="images/":
                                data.append(relative_path)
                            else:
                                if file_path.split("gtFine_")[1] == "labelTrainIds.png":
                                    label.append(relative_path)
        return data, label
