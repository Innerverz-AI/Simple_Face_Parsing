import os
import sys
sys.path.append('./')
# from packages import FaceParser
from main import FaceParser
from PIL import Image
from torchvision import transforms
import cv2
from tqdm import tqdm

faceparser = FaceParser()

celebHQ = "/home/compu/dataset/CelebHQ"
celebHQ_Mask = "/home/leee/data/CelebHQ-mask"
KF_F = "/home/compu/dataset/KF-dataset/KFW/KFW_HR/KFW_HR_F/clean"
KF_F_Mask = "/home/leee/data/KF-mask/F"
KF_M = "/home/compu/dataset/KF-dataset/KFW/KFW_HR/KFW_HR_M/clean"
KF_M_Mask = "/home/leee/data/KF-mask/M"

img_paths = [os.path.join(KF_M, f) for f in os.listdir(KF_M) if f.endswith(".png")]
print(f"* Total {len(img_paths)} images")

transform = transforms.Compose([
    transforms.Resize(1024),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

for ip in tqdm(img_paths):
    # img = Image.open("./assets/000014.jpg").convert("RGB")
    basename = os.path.basename(ip)
    img = Image.open(ip).convert("RGB")

    parsing_map = faceparser.get_face_mask((transform(img).unsqueeze(0).cuda()), do_dilate=True, do_blur=True)
    parsing_map = parsing_map.squeeze().detach().cpu().numpy().transpose([1,2,0])
    # cv2.imwrite("packages/face_parsing/test.jpg", parsing_map*255)
    cv2.imwrite(os.path.join(KF_M_Mask, "m"+basename), parsing_map*255)



