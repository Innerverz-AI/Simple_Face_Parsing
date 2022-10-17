import os
import sys
sys.path.append('./')
from face_parsing.main import FaceParser
from PIL import Image
from torchvision import transforms
import cv2
import glob
from tqdm import tqdm

faceparser = FaceParser()

saveroot = "/media/deep3090/hdd/mask_dataset_re"
saveroot_img = f"{saveroot}/img"
saveroot_fullmask = f"{saveroot}/fullmask"
saveroot_headmask = f"{saveroot}/headmask"
saveroot_facemask = f"{saveroot}/facemask"
saveroot_facemask_wo_ear = f"{saveroot}/facemask_wo_ear"
saveroot_facemask_wo_eyeglasses = f"{saveroot}/facemask_wo_eyeglasses"

os.makedirs(saveroot, exist_ok=True)
os.makedirs(saveroot_img, exist_ok=True)
os.makedirs(saveroot_fullmask, exist_ok=True)
os.makedirs(saveroot_headmask, exist_ok=True)
os.makedirs(saveroot_facemask, exist_ok=True)
os.makedirs(saveroot_facemask_wo_ear, exist_ok=True)
os.makedirs(saveroot_facemask_wo_eyeglasses, exist_ok=True)

ffhqroot = "/media/deep3090/hdd/ffhq70k"

count = 0
ffhq_imgpaths = []
for root, dirs, files in os.walk(ffhqroot):
    for dir in dirs:

        img_paths = glob.glob(f"{root}/{dir}/*.*")
        count += len(img_paths)
        ffhq_imgpaths += img_paths

transform = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

count = 0
for img_path in tqdm(ffhq_imgpaths):
    basename = os.path.splitext(os.path.basename(img_path))[0]
    img = Image.open(img_path).convert("RGB")

    fullmask, headmask, facemask, facemask_wo_ear, facemask_wo_eyeglasses = faceparser.get_face_mask((transform(img).unsqueeze(0).cuda()))

    fullmask = fullmask.squeeze().detach().cpu().numpy().transpose([1,2,0])
    headmask = headmask.squeeze().detach().cpu().numpy().transpose([1,2,0])
    facemask = facemask.squeeze().detach().cpu().numpy().transpose([1,2,0])
    facemask_wo_ear = facemask_wo_ear.squeeze().detach().cpu().numpy().transpose([1,2,0])
    facemask_wo_eyeglasses = facemask_wo_eyeglasses.squeeze().detach().cpu().numpy().transpose([1,2,0])

    img.resize((512,512)).save(f"{saveroot_img}/{basename}.jpg")
    cv2.imwrite(f"{saveroot_fullmask}/{basename}.jpg", fullmask*255)
    cv2.imwrite(f"{saveroot_headmask}/{basename}.jpg", headmask*255)
    cv2.imwrite(f"{saveroot_facemask}/{basename}.jpg", facemask*255)
    cv2.imwrite(f"{saveroot_facemask_wo_ear}/{basename}.jpg", facemask_wo_ear*255)
    cv2.imwrite(f"{saveroot_facemask_wo_eyeglasses}/{basename}.jpg", facemask_wo_eyeglasses*255)
    count += 1



