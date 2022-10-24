import sys
import glob
sys.path.append('./')
from PIL import Image
import random
from torchvision import transforms
import cv2
from libs.bisenet import BiSeNet
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os

from face_parsing.main import FaceParser
GaussianBlur = transforms.GaussianBlur(kernel_size=25, sigma=(0.1, 5))

faceparser = FaceParser()

to_tensor = transforms.Compose([
    transforms.Resize(512),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

ckptroot = "/home/deep3090/workspace/face-parsing/train_result/resnet18/256_512_x2_64_silu/ckpt"
ckpt_files = sorted(os.listdir(ckptroot))
ckpt_num = len(ckpt_files)

dataroot = "/home/deep3090/workspace/face-parsing/assets/test_images/all"
img_paths = []
img_paths = glob.glob(f"{dataroot}/*.*")[:8]
saveroot = "/home/deep3090/workspace/face-parsing/assets/result_images"

results = np.zeros((512*(ckpt_num+2), len(img_paths)*512, 3))
print(results.shape)

for img_idx, img_path in enumerate(img_paths):

    print(img_path)

    img = Image.open(img_path).convert("RGB").resize((512,512))
    parsing_map = faceparser.get_face_mask(to_tensor(img).unsqueeze(0).cuda())[1]

    results[:512, img_idx*512:(img_idx+1)*512, :] = np.array(img)/255
    results[512:1024, img_idx*512:(img_idx+1)*512, :] = parsing_map[0].detach().cpu().numpy().transpose([1,2,0])

for ckpt_idx, ckptfile in enumerate(ckpt_files):
    ckptname = os.path.splitext(ckptfile)[0]

    net = BiSeNet(n_classes=1).cuda().eval()
    ckpt = torch.load(f"{ckptroot}/{ckptfile}", map_location="cuda")
    net.load_state_dict(ckpt['model'], strict=True) 
    savedir = f"{saveroot}/{ckptname}"
    os.makedirs(savedir, exist_ok=True)

    for img_idx, img_path in enumerate(img_paths):

        print(img_path)
        
        img = Image.open(img_path).convert("RGB").resize((512,512))
        parsing_map = net(to_tensor(img).unsqueeze(0).cuda())[0]
        # parsing_map = torch.where(GaussianBlur(parsing_map)>0.5, 1, 0)
        parsing_map = parsing_map.repeat(1,3,1,1).squeeze(0).detach().cpu().numpy().transpose([1,2,0])
        
        results[512*(ckpt_idx+2):512*(ckpt_idx+3), img_idx*512:(img_idx+1)*512, :] = parsing_map[:, :, 1::4]
        img_name = os.path.basename(img_path)
        H, W = results.shape[:2]

cv2.imwrite(f"./testgrid_all_nothres.png", results[:, :, ::-1]*255)



