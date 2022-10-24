import sys
sys.path.append('./')

from old.main import FaceParserOld
from new.main import FaceParser
from PIL import Image

import torch
import torchvision
from torchvision import transforms
import cv2, glob, os

faceparserold = FaceParserOld()
faceparser = FaceParser()
img_paths = sorted(glob.glob("./assets/test_image/*.*"))

transform_old = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

transform_new = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

imgs_list = [[],[],[],[],[],[],[],[],[]]
tensor_list = []

for img_path in img_paths:
    img_name = os.path.basename(img_path)
    img = Image.open(img_path).convert("RGB")
    img_for_old = transform_old(img).unsqueeze(0).cuda() # pixel valu [0, 1]
    img_for_new = transform_new(img).unsqueeze(0).cuda() # pixel valu [-1, 1]

    full_mask_old, head_mask_old, face_mask_old, inner_face_mask_old = faceparserold.get_four_masks(img_for_old)
    full_mask, head_mask, face_mask, inner_face_mask = faceparser.get_four_masks(img_for_new)

    imgs_list[0].append(img_for_old)
    for i, data in enumerate([full_mask_old, head_mask_old, face_mask_old, inner_face_mask_old, full_mask, head_mask, face_mask, inner_face_mask]):
        imgs_list[i+1].append(data.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1))

for i in range(9):
    tensor_list.append(torch.cat(imgs_list[i], dim=0))

grid_rows = []
for tensor in tensor_list:
    grid_row = torchvision.utils.make_grid(tensor, nrow=tensor.shape[0]) 
    grid_rows.append(grid_row)

grid = torch.cat(grid_rows, dim=1)
grid = grid.detach().cpu().numpy().transpose([1,2,0])[:, :, ::-1] * 255
cv2.imwrite(f"./assets/grid_image.png", grid)


