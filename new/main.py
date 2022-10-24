import torch
import torch.nn as nn
import torch.nn.functional as F
from new.model import BiSeNet
import torchvision.transforms as transforms
GaussianBlur = transforms.GaussianBlur(kernel_size=25, sigma=(0.1, 5))

class FaceParser(nn.Module):
    def __init__(self, file_PATH = '/home/deep3090/workspace/Simple_Face_Parsing/train_result/sixth_try/ckpt/BiSe_new.pt'):
    # def __init__(self, file_PATH = './new/ckpt/BiSe_10000.pt'):
        super(FaceParser, self).__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.parsing_net = BiSeNet(n_classes=4).to(device)
        ckpt = torch.load(file_PATH, map_location=device)
        self.parsing_net.load_state_dict(ckpt['model'])
        for param in self.parsing_net.parameters():
            param.requires_grad = False
        self.parsing_net.eval()
        del ckpt

    def get_four_masks(self, tensor_image):
        tensor_image = F.interpolate(tensor_image, (512, 512), mode='bilinear')
        parsing = self.parsing_net(tensor_image)
        parsing = torch.where(GaussianBlur(parsing)>0.5, 1, 0)
        parsing = parsing.squeeze(0)
        full_mask = parsing[0]
        head_mask = parsing[1]
        face_mask = parsing[2]
        inner_face_mask = parsing[3]

        return full_mask, head_mask, face_mask, inner_face_mask
