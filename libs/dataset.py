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

class  FaceDatasetWithOcclusion(Dataset):
    def __init__(self):
        super(FaceDatasetWithOcclusion, self).__init__()

        loadroot = ["/media/deep3090/hdd/mask_dataset_re"]
        self.img_root_path = f"{loadroot[0]}/img"        
        self.img_paths = glob.glob(f"{self.img_root_path}/*.*")

        self.loadroot_fullmask = f"{loadroot[0]}/fullmask"
        self.loadroot_headmask = f"{loadroot[0]}/headmask"
        self.loadroot_facemask = f"{loadroot[0]}/facemask"
        self.loadroot_facemask_wo_ear = f"{loadroot[0]}/facemask_wo_ear"
        # self.loadroot_facemask_wo_eyeglasses = f"{loadroot[0]}/facemask_wo_eyeglasses"

        self.imgroot = "/home/deep3090/workspace/dataset/PASCAL_VOC_2012/JPEGImages"
        segroot = "/home/deep3090/workspace/dataset/PASCAL_VOC_2012/SegmentationClass"
        self.segpaths = glob.glob(f"{segroot}/*.*")

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
        # mask_path5 = os.path.join(self.loadroot_facemask_wo_eyeglasses, basename)

        face = cv2.imread(img_path) 

        masks = []
        masks.append(cv2.imread(mask_path1, 0).reshape(1, 512, 512))
        masks.append(cv2.imread(mask_path2, 0).reshape(1, 512, 512))
        masks.append(cv2.imread(mask_path3, 0).reshape(1, 512, 512))
        masks.append(cv2.imread(mask_path4, 0).reshape(1, 512, 512))
        # masks.append(cv2.imread(mask_path5, 0).reshape(1, 512, 512))

        if random.random() < 0.25:
            mask = np.concatenate(masks, axis=0)
            mask = torch.from_numpy(mask/255).float()
            face = self.to_tensor(Image.fromarray(face[:, :, ::-1]))

        else:
            # occlusion img, seg
            segpath = random.choice(self.segpaths)
            imgname = os.path.basename(segpath)[:-4]
            imgpath = f"{self.imgroot}/{imgname}.jpg"

            seg = np.zeros((512,512))
            img = np.zeros((512,512,3))

            x, y = np.random.choice(256, 2)
            seg[x:x+256, y:y+256] = cv2.flip(cv2.resize(cv2.imread(segpath, 0), (256, 256)), 0) # load seg mask as a grayscale because it is 3ch segmentation
            img[x:x+256, y:y+256, :] = cv2.flip(cv2.resize(cv2.imread(imgpath), (256, 256)), 0)

            seg = np.where(seg>0, 1, 0)
            seg = seg[:, :, None].repeat(3, axis=2)
            seg = cv2.erode((seg*255).astype(np.uint8), kernel=np.ones((3,3)), iterations=3)/255
            segimg = img * seg

            k = int(np.random.choice(20, 1)) + 1
            blend = face * (1-seg) + segimg
            blend = cv2.blur((blend).astype(np.uint8), (k,k))
            seg_ = cv2.blur((seg*255).astype(np.uint8), (k,k))/255
            face = face * (1-seg_) + blend * seg_
            seg_ = seg_[:, :, 0].reshape(1, 512, 512)

            occlusion_masks = []
            for i in range(4):
                occlusion_mask = masks[i]/255 - seg_
                occlusion_mask = np.where(occlusion_mask>0.4, 1, 0)
                occlusion_masks.append(occlusion_mask)
            occlusion_masks = np.concatenate(occlusion_masks, axis=0)
            mask = torch.from_numpy(occlusion_masks).float()
            face = self.to_tensor(Image.fromarray(face[:, :, ::-1].astype(np.uint8)))

        return face, mask

    def __len__(self):
        return len(self.img_paths)