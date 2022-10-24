import os
import sys
sys.path.append('./')
from face_parsing.main import FaceParser
from PIL import Image
from torchvision import transforms
import cv2
import glob
from tqdm import tqdm
import numpy as np
import random

faceparser = FaceParser()

loadroot = "/media/deep3090/hdd/mask_dataset"
loadroot_img = f"{loadroot}/img"
loadroot_fullmask = f"{loadroot}/fullmask"
loadroot_headmask = f"{loadroot}/headmask"
loadroot_facemask = f"{loadroot}/facemask"
loadroot_facemask_wo_ear = f"{loadroot}/facemask_wo_ear"

saveroot = "/media/deep3090/hdd/occlusion_mask_dataset"
saveroot_img = f"{saveroot}/img"
saveroot_fullmask = f"{saveroot}/fullmask"
saveroot_headmask = f"{saveroot}/headmask"
saveroot_facemask = f"{saveroot}/facemask"
saveroot_facemask_wo_ear = f"{saveroot}/facemask_wo_ear"
saveroots = [saveroot_fullmask, saveroot_headmask, saveroot_facemask, saveroot_facemask_wo_ear]

os.makedirs(saveroot, exist_ok=True)
os.makedirs(saveroot_img, exist_ok=True)
os.makedirs(saveroot_fullmask, exist_ok=True)
os.makedirs(saveroot_headmask, exist_ok=True)
os.makedirs(saveroot_facemask, exist_ok=True)
os.makedirs(saveroot_facemask_wo_ear, exist_ok=True)

# ffhqroot = "/media/deep3090/hdd/ffhq70k"

# count = 0
# ffhq_imgpaths = []
# for root, dirs, files in os.walk(ffhqroot):
#     for dir in dirs:

#         img_paths = glob.glob(f"{root}/{dir}/*.*")
#         count += len(img_paths)
#         ffhq_imgpaths += img_paths

# transform = transforms.Compose([
#     transforms.Resize(512),
#     transforms.ToTensor(),
#     # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
# ])

imgroot = "/home/deep3090/workspace/dataset/PASCAL_VOC_2012/JPEGImages"
segroot = "/home/deep3090/workspace/dataset/PASCAL_VOC_2012/SegmentationClass"
segpaths = glob.glob(f"{segroot}/*.*")

img_paths = sorted(glob.glob(f"{loadroot_img}/*.*"))
for img_path in tqdm(img_paths):
    
    # face, mask
    basename = os.path.basename(img_path)
    face = cv2.imread(img_path)

    fullmask = cv2.imread(f"{loadroot_fullmask}/{basename}")/255
    headmask = cv2.imread(f"{loadroot_headmask}/{basename}")/255
    facemask = cv2.imread(f"{loadroot_facemask}/{basename}")/255
    facemask_wo_ear = cv2.imread(f"{loadroot_facemask_wo_ear}/{basename}")/255
    masks = [fullmask, headmask, facemask, facemask_wo_ear]

    # occlusion img, seg
    segpath = random.choice(segpaths)
    imgname = os.path.basename(segpath)[:-4]
    imgpath = f"{imgroot}/{imgname}.jpg"

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
    occlusion_face = face * (1-seg_) + blend * seg_
    cv2.imwrite(f"{saveroot_img}/{basename}", occlusion_face)

    for i in range(4):
        occlusion_mask = masks[i] - seg_
        occlusion_mask = np.where(occlusion_mask>0.4, 1, 0)
        cv2.imwrite(f"{saveroots[i]}/{basename}", occlusion_mask*255)



