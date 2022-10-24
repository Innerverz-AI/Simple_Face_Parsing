import torch
import torch.nn as nn
import torch.nn.functional as F
from old.model import BiSeNet
from torchvision import transforms

# index = [
# 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 
# 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 
# 11 'mouth', 12 'u_lip', 13 'l_lip', 
# 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

class FaceParserOld(nn.Module):
    def __init__(self, file_PATH = './old/ckpt/faceparser.pth'):
        super(FaceParserOld, self).__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.parsing_net = BiSeNet(n_classes=19).to(device)
        ckpt = torch.load(file_PATH, map_location=device)
        self.parsing_net.load_state_dict(ckpt)
        for param in self.parsing_net.parameters():
            param.requires_grad = False
        self.parsing_net.eval()
        del ckpt

    def forward(self, tensor_image):
        tensor_image = F.interpolate(tensor_image, (512, 512), mode='bilinear')
        parsing = self.parsing_net(tensor_image).max(1)[1] # index map
        return parsing # [B, H, W]
        
    def get_four_masks(self, tensor_image):
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

        # facemask_wo_eyeglasses = facemask_wo_ear.clone()
        # facemask_wo_eyeglasses-= torch.where(parsing==6, 1, 0)

        return fullmask.squeeze(), headmask.squeeze(), facemask.squeeze(), facemask_wo_ear.squeeze()
            
            