import sys
sys.path.append('./')
sys.path.append('./packages')
from face_parsing.main import FaceParser
from PIL import Image
from torchvision import transforms
import cv2, glob, os
import numpy as np
import torch.nn.functional as F
kernel = np.ones((3, 3), np.uint8)

faceparser = FaceParser()
img_paths = glob.glob("/home/compu/AttributeRegressor/test/*.*")[:20]
for img_path in ["/home/deep3090/workspace/face-parsing/assets/f_0020.png"]:
    img_name = os.path.basename(img_path)
    img = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    parsing_map = faceparser.get_face_mask((transform(img).unsqueeze(0).cuda()))[1].float()
    parsing_map_ = parsing_map.squeeze().detach().cpu().numpy().transpose([1,2,0])
    cv2.imwrite(f"./test.png", 255*parsing_map_)

    parsing_map = F.interpolate(parsing_map, (64, 64), mode = 'bilinear')
    parsing_map = parsing_map.squeeze().detach().cpu().numpy().transpose([1,2,0])
    parsing_map = cv2.resize(parsing_map*255, (512,512))

    cv2.imwrite(f"./test2.png", parsing_map)

    parsing_map = cv2.erode(parsing_map, kernel, iterations=2)
    parsing_map = cv2.dilate(parsing_map, kernel, iterations=2)

    cv2.imwrite(f"./test3.png", parsing_map)


