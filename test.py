import sys
sys.path.append('./')
# from packages import FaceParser
from main import FaceParser
from PIL import Image
from torchvision import transforms
import cv2

faceparser = FaceParser()

img = Image.open("./assets/000014.jpg").convert("RGB")

transform = transforms.Compose([
    transforms.Resize(1024),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

parsing_map = faceparser.get_face_mask((transform(img).unsqueeze(0).cuda()), do_dilate=False, do_blur=True)
parsing_map = parsing_map.squeeze().detach().cpu().numpy().transpose([1,2,0])
# cv2.imwrite("packages/face_parsing/test.jpg", parsing_map*255)
cv2.imwrite("./results/000014.jpg", parsing_map*255)



