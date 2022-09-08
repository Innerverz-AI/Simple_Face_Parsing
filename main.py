import torch
import torch.nn as nn
import torch.nn.functional as F
# from packages.face_parsing.model import BiSeNet
from models.bisenet import BiSeNet
from torchvision import transforms

# index = [
# 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 
# 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 
# 11 'mouth', 12 'u_lip', 13 'l_lip', 
# 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

class FaceParser(nn.Module):
    def __init__(self, file_PATH = './ckpt/faceparser.pth'):
        super(FaceParser, self).__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.parsing_net = BiSeNet(n_classes=19).to(device)
        ckpt = torch.load(file_PATH, map_location=device)
        self.parsing_net.load_state_dict(ckpt)
        for param in self.parsing_net.parameters():
            param.requires_grad = False
        self.parsing_net.eval()
        del ckpt

        self.GaussianBlur = transforms.GaussianBlur(kernel_size=25, sigma=(0.1, 5))
        self.kernel = torch.ones((1,1,5,5), device="cuda")

    def get_face_mask(self, tensor_image, do_dilate=False, do_blur=True):
        H, W = tensor_image.size()[2:]
        tensor_image = F.interpolate(tensor_image, (512, 512), mode='bilinear')
        # res = self.parsing_net(tensor_image) # [1, 19, 512, 512]
        parsing = self.parsing_net(tensor_image).max(1)[1]
        mask = torch.where(parsing<14, 1, 0)
        mask-= torch.where(parsing==0, 1, 0)
        mask-= torch.where(parsing==7, 1, 0)
        mask-= torch.where(parsing==8, 1, 0)
        mask-= torch.where(parsing==9, 1, 0) # [1, 512, 512]
        mask = mask.unsqueeze(1).float()
    
        if do_dilate:
            mask = torch.clamp(F.conv2d(mask, self.kernel, padding=(2, 2)), 0, 1)

        if do_blur:
            mask = self.GaussianBlur(mask)

        mask = F.interpolate(mask, (512, 512), mode='bilinear')
        return mask.repeat(1,3,1,1)

            
    def get_eye_mask(self, tensor_image):
        tensor_image = F.interpolate(tensor_image, (512, 512))
        parsing = self.parsing_net(tensor_image).max(1)[1]
        mask = torch.where(parsing==4, 1, 0) + torch.where(parsing==5, 1, 0)
        mask = mask.unsqueeze(1).float()
    
        return mask.repeat(1,3,1,1)

            