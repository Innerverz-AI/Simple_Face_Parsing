import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from libs.bisenet import BiSeNet

import os
import cv2
from tqdm import tqdm
from PIL import Image

"""
index:
1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 
7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 
11 'mouth', 12 'u_lip', 13 'l_lip', 
14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat'
"""

class FaceParser(nn.Module):
    def __init__(self, baseline):
        super(FaceParser, self).__init__()
        self.parsing_net = None
    
        # Choose model
        if baseline == "bisenet":
            self.parsing_net = BiSeNet(n_classes=19)
            ckpt_path = "ckpt/BiSe_10000.pt"
        else:
            raise Exception("Invalid baseline")
        
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cuda")
        self.parsing_net.load_state_dict(ckpt["model"])
        for param in self.parsing_net.parameters():
            param.requires_grad = False
        self.parsing_net.cuda()
        self.parsing_net.eval()
        del ckpt

    def get_face_mask(self, image):
        H, W = image.size()[2:]
        image = F.interpolate(image, (512, 512), mode='bilinear')
        mask, _, _ = self.parsing_net(image)
        mask = mask.unsqueeze(1).float()
        mask = F.interpolate(mask, (512, 512), mode='bilinear')
        return mask.repeat(1,3,1,1)


def inference(baseline, load_path, save_path):
    parser = FaceParser(baseline)
    
    img_paths = [os.path.join(load_path, f) for f in os.listdir(load_path) if f.endswith(".png") or f.endswith(".jpg")]
    print(f"* Total {len(img_paths)} images")

    transform = transforms.Compose([
        transforms.Resize(1024),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    for ip in tqdm(img_paths):
        basename = os.path.basename(ip)
        img = Image.open(ip).convert("RGB")

        parsing_map = parser.get_face_mask((transform(img).unsqueeze(0).cuda()))
        parsing_map = parsing_map.squeeze().detach().cpu().numpy().transpose([1,2,0])

        cv2.imwrite(os.path.join(save_path, "m"+basename), parsing_map*255)

if __name__ == "__main__":
    baseline = "bisenet"

    load_path = "./assets"
    save_path = "./results"
    os.makedirs(save_path, exist_ok=True)

    inference(baseline, load_path, save_path)