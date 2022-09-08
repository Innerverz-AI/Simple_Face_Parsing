import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from libs.transform import *

import os
import numpy as np
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, data_roots, crop_size=(640, 480), mode='train'):
        super(FaceDataset, self).__init__()

        self.img_root_path, self.mask_root_path = data_roots
        self.mode = mode
        
        self.imgs = [f for f in os.listdir(self.img_root_path) if f.endswith(".png") or f.endswith(".jpg")]

        # transforms
        # self.color_jitter = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.augmentation = Compose([
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            Resize(crop_size) # RandomCrop(crop_size) # Y. random crop instead of resize
        ])
         

    def __getitem__(self, idx):
        basename = self.imgs[idx]
        img_path = os.path.join(self.img_root_path, basename)
        mask_path = os.path.join(self.mask_root_path, "m"+basename)[:-4] + ".jpg"
        assert os.path.isfile(mask_path)

        img = Image.open(img_path).convert("RGB") 
        mask = Image.open(mask_path).convert("L") # Y. grayscale; original repo uses "P"

        # print(np.array(img).shape, np.array(mask).shape) # (1024, 1024, 3) (512, 512)

        if self.mode == 'train':
            # img = self.color_jitter(img) # Y. jittering on images only (not for binary masks)
            pair = dict(im=img, lb=mask) 
            pair = self.augmentation(pair)
            img, mask = pair['im'], pair['lb']
        
        img = self.to_tensor(img)
        mask = transforms.ToTensor()(mask)

        # print(img.shape, mask.shape) # [3, 448, 448], [1, 448, 448]
        # mask = np.array(mask).astype(np.int64)[np.newaxis, :]
        return img, mask


    def __len__(self):
        return len(self.imgs)