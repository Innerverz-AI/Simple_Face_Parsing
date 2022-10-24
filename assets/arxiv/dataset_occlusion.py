import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from libs.transform import *
import cv2
import os
import numpy as np
from PIL import Image
import glob
import random

class  FaceDataset(Dataset):
    def __init__(self):
        super(FaceDataset, self).__init__()

        loadroot = ["/media/deep3090/hdd/mask_dataset", "/media/deep3090/hdd/occlusion_mask_dataset"]
        self.img_root_path = f"{loadroot[0]}/img"        
        self.img_paths = glob.glob(f"{self.img_root_path}/*.*")

        self.loadroot_fullmask = f"{loadroot[0]}/fullmask"
        self.loadroot_headmask = f"{loadroot[0]}/headmask"
        self.loadroot_facemask = f"{loadroot[0]}/facemask"
        self.loadroot_facemask_wo_ear = f"{loadroot[0]}/facemask_wo_ear"

        self.img_root_path_occlusion = f"{loadroot[1]}/img"        
        self.img_paths_occlusion = glob.glob(f"{self.img_root_path_occlusion}/*.*")

        self.loadroot_fullmask_occlusion = f"{loadroot[1]}/fullmask"
        self.loadroot_headmask_occlusion = f"{loadroot[1]}/headmask"
        self.loadroot_facemask_occlusion = f"{loadroot[1]}/facemask"
        self.loadroot_facemask_wo_ear_occlusion = f"{loadroot[1]}/facemask_wo_ear"

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
        
        if random.random() > 0.5:
            img_path = os.path.join(self.img_root_path, basename)
            mask_path1 = os.path.join(self.loadroot_fullmask, basename)
            mask_path2 = os.path.join(self.loadroot_headmask, basename)
            mask_path3 = os.path.join(self.loadroot_facemask, basename)
            mask_path4 = os.path.join(self.loadroot_facemask_wo_ear, basename)

        else:
            img_path = os.path.join(self.img_root_path_occlusion, basename)
            mask_path1 = os.path.join(self.loadroot_fullmask_occlusion, basename)
            mask_path2 = os.path.join(self.loadroot_headmask_occlusion, basename)
            mask_path3 = os.path.join(self.loadroot_facemask_occlusion, basename)
            mask_path4 = os.path.join(self.loadroot_facemask_wo_ear_occlusion, basename)


        img = Image.open(img_path).convert("RGB") 
        img = self.to_tensor(img)

        mask1 = cv2.imread(mask_path1, 0).reshape(1, 512, 512)
        mask2 = cv2.imread(mask_path2, 0).reshape(1, 512, 512)
        mask3 = cv2.imread(mask_path3, 0).reshape(1, 512, 512)
        mask4 = cv2.imread(mask_path4, 0).reshape(1, 512, 512)
        mask = np.concatenate([mask1, mask2, mask3, mask4], axis=0)
        mask = torch.from_numpy(mask/255).float()

        return img, mask

    def __len__(self):
        return len(self.img_paths)