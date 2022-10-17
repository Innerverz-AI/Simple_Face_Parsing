import sys
import glob
sys.path.append('./')
from PIL import Image
import random
from torchvision import transforms
import cv2
from models.bisenet import BiSeNet
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os

from face_parsing.main import FaceParser

from torchvision import transforms
GaussianBlur = transforms.GaussianBlur(kernel_size=25, sigma=(0.1, 5))
faceparser = FaceParser()

to_tensor = transforms.Compose([
    transforms.Resize(512),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataroot = "/home/deep3090/workspace/face-parsing/assets/test_images/all"
img_paths = []
img_paths = glob.glob(f"{dataroot}/*.*")[:8]

saveroot = "/home/deep3090/workspace/face-parsing/assets/result_images"

results = np.zeros((512*10, len(img_paths)*512, 3))
print(results.shape)

for img_idx, img_path in enumerate(img_paths):

    print(img_path)

    img = Image.open(img_path).convert("RGB").resize((512,512))
    parsing_map = faceparser.get_face_mask(to_tensor(img).unsqueeze(0).cuda())

    results[:512, img_idx*512:(img_idx+1)*512, :] = np.array(img)/255
    results[512:512*2, img_idx*512:(img_idx+1)*512, :] = parsing_map[0][0].detach().cpu().numpy().transpose([1,2,0])
    results[512*2:512*3, img_idx*512:(img_idx+1)*512, :] = parsing_map[1][0].detach().cpu().numpy().transpose([1,2,0])
    results[512*3:512*4, img_idx*512:(img_idx+1)*512, :] = parsing_map[2][0].detach().cpu().numpy().transpose([1,2,0])
    results[512*4:512*5, img_idx*512:(img_idx+1)*512, :] = parsing_map[3][0].detach().cpu().numpy().transpose([1,2,0])
    results[512*5:512*6, img_idx*512:(img_idx+1)*512, :] = np.array(img)/255


net = BiSeNet(n_classes=4).cuda().eval()
ckpt = torch.load("/home/deep3090/workspace/face-parsing/train_result/occlusion/ckpt/BiSe_70000.pt", map_location="cuda")
net.load_state_dict(ckpt['model'], strict=True) 

for img_idx, img_path in enumerate(img_paths):

    print(img_path)
    
    img = Image.open(img_path).convert("RGB").resize((512,512))
    parsing_map0, parsing_map1, parsing_map2, parsing_map3 = net(to_tensor(img).unsqueeze(0).cuda())[0]
    # parsing_map = F.interpolate(parsing_map, (512,512), mode='bilinear')
    # parsing_map = torch.where(parsing_map>0.5, 1, 0)
    # parsing_map = GaussianBlur(parsing_map)

    parsing_map0 = torch.where(parsing_map0>0.5, 1, 0)
    parsing_map1 = torch.where(parsing_map1>0.5, 1, 0)
    parsing_map2 = torch.where(parsing_map2>0.5, 1, 0)
    parsing_map3 = torch.where(parsing_map3>0.5, 1, 0)

    parsing_map0 = parsing_map0.repeat(1,3,1,1).squeeze(0).detach().cpu().numpy().transpose([1,2,0])
    parsing_map1 = parsing_map1.repeat(1,3,1,1).squeeze(0).detach().cpu().numpy().transpose([1,2,0])
    parsing_map2 = parsing_map2.repeat(1,3,1,1).squeeze(0).detach().cpu().numpy().transpose([1,2,0])
    parsing_map3 = parsing_map3.repeat(1,3,1,1).squeeze(0).detach().cpu().numpy().transpose([1,2,0])
    results[512*6:512*7, img_idx*512:(img_idx+1)*512, :] = parsing_map0
    results[512*7:512*8, img_idx*512:(img_idx+1)*512, :] = parsing_map1
    results[512*8:512*9, img_idx*512:(img_idx+1)*512, :] = parsing_map2
    results[512*9:512*10, img_idx*512:(img_idx+1)*512, :] = parsing_map3
    img_name = os.path.basename(img_path)
    H, W = results.shape[:2]
cv2.imwrite(f"./grid_all_types4.png", results[:, :, ::-1]*255)


