import torch
import torch.nn as nn
import torch.nn.functional as F
from face_parsing.model import BiSeNet
from torchvision import transforms


"""
This code borrows heavily from github repository below
zllrunning, face-parsing.PyTorch (2019)
https://github.com/zllrunning/face-parsing.PyTorch
"""

# index = [
# 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 
# 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 
# 11 'mouth', 12 'u_lip', 13 'l_lip', 
# 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

class FaceParser(nn.Module):
    def __init__(self, file_PATH = './face_parsing/ckpt/faceparser.pth'):
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

    def get_face_mask(self, tensor_image, do_dilate=False, do_blur=False):
        H, W = tensor_image.size()[2:]
        tensor_image = F.interpolate(tensor_image, (512, 512), mode='bilinear')
        parsing = self.parsing_net(tensor_image).max(1)[1]
        fullmask = torch.where(parsing>0, 1, 0)

        headmask = fullmask.clone()
        headmask-= torch.where(parsing==18, 1, 0)
        headmask-= torch.where(parsing==16, 1, 0)
        headmask-= torch.where(parsing==15, 1, 0)
        headmask-= torch.where(parsing==14, 1, 0)

        facemask = headmask.clone()
        facemask -= torch.where(parsing==17, 1, 0)

        facemask_wo_ear = facemask.clone()
        facemask_wo_ear-= torch.where(parsing==7, 1, 0)
        facemask_wo_ear-= torch.where(parsing==8, 1, 0)
        facemask_wo_ear-= torch.where(parsing==9, 1, 0)

        facemask_wo_eyeglasses = facemask_wo_ear.clone()
        facemask_wo_eyeglasses-= torch.where(parsing==6, 1, 0)

        return fullmask.repeat(1,3,1,1), headmask.repeat(1,3,1,1), facemask.repeat(1,3,1,1), facemask_wo_ear.repeat(1,3,1,1), facemask_wo_eyeglasses.repeat(1,3,1,1)

    def get_4_mask(self, tensor_image, do_dilate=False, do_blur=False):
        H, W = tensor_image.size()[2:]
        tensor_image = F.interpolate(tensor_image, (512, 512), mode='bilinear')
        parsing = self.parsing_net(tensor_image).max(1)[1]
        fullmask = torch.where(parsing>0, 1, 0)

        headmask = fullmask.clone()
        headmask-= torch.where(parsing==18, 1, 0)
        headmask-= torch.where(parsing==16, 1, 0)
        headmask-= torch.where(parsing==15, 1, 0)
        headmask-= torch.where(parsing==14, 1, 0)

        facemask = headmask.clone()
        facemask -= torch.where(parsing==17, 1, 0)

        facemask_wo_ear = facemask.clone()
        facemask_wo_ear-= torch.where(parsing==7, 1, 0)
        facemask_wo_ear-= torch.where(parsing==8, 1, 0)
        facemask_wo_ear-= torch.where(parsing==9, 1, 0)

        facemask = F.interpolate(facemask.unsqueeze(0).float(), (512, 512), mode='bilinear')
        return fullmask.repeat(1,3,1,1), headmask.repeat(1,3,1,1), facemask.repeat(1,3,1,1), facemask_wo_ear.repeat(1,3,1,1)
            
    def get_eye_mask(self, tensor_image):
        tensor_image = F.interpolate(tensor_image, (512, 512))
        parsing = self.parsing_net(tensor_image).max(1)[1]
        mask = torch.where(parsing==4, 1, 0) + torch.where(parsing==5, 1, 0)
        mask = mask.unsqueeze(1).float()
    
        return mask.repeat(1,3,1,1)

            