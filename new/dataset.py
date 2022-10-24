import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import cv2
import os
import numpy as np
from PIL import Image
import glob
import random

class FaceDataset(Dataset):
    def __init__(self, loadroot):
        super(FaceDataset, self).__init__()

        self.img_root_path = f"{loadroot}/img"
        self.img_paths = glob.glob(f"{self.img_root_path}/*.*")

        self.loadroot_fullmask = f"{loadroot}/fullmask"
        self.loadroot_headmask = f"{loadroot}/headmask"
        self.loadroot_facemask = f"{loadroot}/facemask"
        self.loadroot_facemask_wo_ear = f"{loadroot}/facemask_wo_ear"

        # transforms
        self.to_tensor = transforms.Compose([
            transforms.Resize(512),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        print(f"dataset has loaded: {len(self.img_paths)} images")

    def __getitem__(self, idx):

        basename = os.path.basename(self.img_paths[idx])
        
        img_path = os.path.join(self.img_root_path, basename)
        mask_path1 = os.path.join(self.loadroot_fullmask, basename)
        mask_path2 = os.path.join(self.loadroot_headmask, basename)
        mask_path3 = os.path.join(self.loadroot_facemask, basename)
        mask_path4 = os.path.join(self.loadroot_facemask_wo_ear, basename)

        face = cv2.imread(img_path) 

        masks = []
        masks.append(cv2.imread(mask_path1, 0).reshape(1, 512, 512))
        masks.append(cv2.imread(mask_path2, 0).reshape(1, 512, 512))
        masks.append(cv2.imread(mask_path3, 0).reshape(1, 512, 512))
        masks.append(cv2.imread(mask_path4, 0).reshape(1, 512, 512))

        mask = np.concatenate(masks, axis=0)
        mask = torch.from_numpy(mask/255).float()
        face = self.to_tensor(Image.fromarray(face[:, :, ::-1]))

        return face, mask

    def __len__(self):
        return len(self.img_paths)