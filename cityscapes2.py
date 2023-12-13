from torch.utils.data import Dataset
import torch
from torchvision import transforms
from collections import namedtuple
import os
from PIL import Image
import numpy as np

def pil_loader(p, mode):
    with open(p, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)

def one_hot_it(label, label_info):
    semantic_map = np.zeros((19, label.shape[1], label.shape[2]))

    for index, info in enumerate(label_info):
        color = info[:3].reshape(3, 1, 1)
        equality = np.all(label == color, axis=0)
     
        semantic_map[index % 19][equality] = 1

    return torch.tensor(semantic_map)

CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

train_id_to_color = [list(c.color)+[c.train_id] for c in classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color.append([0, 0, 0, 20])
train_id_to_color = np.array(train_id_to_color)

# mode = train or val, path = "/content/Cityscapes/Cityspaces/"
class CityScapes(Dataset):
    def init(self, mode):
        super(CityScapes, self).init()
        self.path = "/content/Cityscapes/Cityspaces/"
        self.mode = mode
        self.data, self.label = self.data_loader()
        self.preprocess = transforms.Compose([
            transforms.PILToTensor(),                 # Converte l'immagine in un tensore
        ])

    

    def getitem(self, idx):
        image = pil_loader(self.path+self.data[idx], 'RGB')
        
        label = pil_loader(self.path+self.label[idx], 'RGB')
    
        tensor_image = self.preprocess(image)

        tensor_label = self.preprocess(label)
        t = one_hot_it(tensor_label, train_id_to_color)
 
        
    
    
        if self.mode == 'train':
          tensor_image = tensor_image.half()
        if self.mode == 'val':
          tensor_image = tensor_image.float()
        
        return tensor_image, t

    def len(self):
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
                                if file_path.split("gtFine_")[1] == "color.png":
                                    label.append(relative_path)
        return sorted(data), sorted(label)