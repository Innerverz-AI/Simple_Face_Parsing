import os, cv2, glob, random
import numpy as np

imgroot = "/home/deep3090/workspace/dataset/PASCAL_VOC_2012/JPEGImages"
segroot = "/home/deep3090/workspace/dataset/PASCAL_VOC_2012/SegmentationClass"

segpaths = glob.glob(f"{segroot}/*.*")
segpaths = random.sample(segpaths, 8)

faceroot = "/media/deep3090/hdd/mask_dataset/img"
maskroot = "/media/deep3090/hdd/mask_dataset/facemask"
facepaths = glob.glob(f"{faceroot}/*.*")
facepaths = random.sample(facepaths, 8)

cols = []
for segpath, facepath in zip(segpaths, facepaths):


    # img, seg

    imgname = os.path.basename(segpath)[:-4]
    imgpath = f"{imgroot}/{imgname}.jpg"

    seg = np.zeros((512,512))
    img = np.zeros((512,512,3))

    x, y = np.random.choice(256, 2)
    seg[x:x+256, y:y+256] = cv2.flip(cv2.resize(cv2.imread(segpath, 0), (256, 256)), 0)
    img[x:x+256, y:y+256, :] = cv2.flip(cv2.resize(cv2.imread(imgpath), (256, 256)), 0)

    seg = np.where(seg>0, 1, 0)
    seg = seg[:, :, None].repeat(3, axis=2)
    seg = cv2.erode((seg*255).astype(np.uint8), kernel=np.ones((3,3)), iterations=3)/255
    segimg = img * seg

    # face, mask

    facename = os.path.basename(facepath)
    maskpath = f"{maskroot}/{facename}"

    face = cv2.resize(cv2.imread(facepath), (512, 512))
    mask = cv2.resize(cv2.imread(maskpath), (512, 512))/255

    # run

    k = int(np.random.choice(20, 1)) + 1

    blend = face * (1-seg) + segimg
    blend = cv2.blur((blend).astype(np.uint8), (k,k))
    seg_ = cv2.blur((seg*255).astype(np.uint8), (k,k))/255
    occlusion_mask = mask - seg_
    occlusion_mask_ = np.where(occlusion_mask>0.25, 1, 0)

    occlusion_face = face * (1-seg_) + blend * seg_

    cv2.imwrite("o_face.jpg", occlusion_face)
    cv2.imwrite("o_mask.jpg", occlusion_mask*255)

    col = np.concatenate([face, mask*255, segimg, seg*255, blend, occlusion_face, occlusion_mask*255, occlusion_mask_*255, occlusion_mask_*255/2+occlusion_face/2], axis=0)
    cols.append(col)

grid = np.concatenate(cols, axis=1)
cv2.imwrite("grid.png", grid)







    

