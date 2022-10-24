import sys
sys.path.append('./')

from new.main import FaceParser
from PIL import Image

import torch
import torchvision
from torchvision import transforms
import cv2, glob, os

faceparser = FaceParser()
img_path = "./assets/test_image/001_face.png"

transform_new = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

img_name = os.path.basename(img_path)
img = Image.open(img_path).convert("RGB")
img_for_new = transform_new(img).unsqueeze(0).cuda() # pixel valu [-1, 1]

full_mask, head_mask, face_mask, inner_face_mask = faceparser.get_four_masks(img_for_new)
tensor = []
tensor.append(img_for_new*0.5+0.5)
for data in [full_mask, head_mask, face_mask, inner_face_mask]:
    tensor.append(data.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1))
tensor = torch.cat(tensor, dim=0)

grid = torchvision.utils.make_grid(tensor, nrow=tensor.shape[0])
grid = grid.detach().cpu().numpy().transpose([1,2,0])[:, :, ::-1] * 255
cv2.imwrite("./assets/grid_single.png", grid)

