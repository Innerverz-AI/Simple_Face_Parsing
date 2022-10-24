import sys
sys.path.append("./")

from new.dataset import FaceDataset 
from new.model import BiSeNet
from new import utils

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import torchvision.transforms as transforms
GaussianBlur = transforms.GaussianBlur(kernel_size=25, sigma=(0.1, 5))

def train(cfg):
    # Dataset
    dataset = FaceDataset(cfg["dataroot"])
    trn_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - trn_size
    trn_dataset, val_dataset = random_split(dataset, [trn_size, val_size])

    # Dataloader
    trn_data_loader = DataLoader(trn_dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_data_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
    
    # Model
    net = BiSeNet(n_classes=cfg["num_classes"]).cuda().train()
    ckpt = torch.load("./old/ckpt/faceparser.pth", map_location="cuda")
    net.load_state_dict(ckpt, strict=False)

    # Loss function
    criterion = nn.BCELoss()
    optim = torch.optim.SGD(net.parameters(), lr=0.02)

    # Train loop
    run_desc = cfg["run_id"]
    max_step = cfg["max_step"]
    log_step = cfg["log_step"]
    val_step = cfg["val_step"]
    ckpt_step = cfg["ckpt_step"]

    loss_dict = {}
    trn_data_iterator = iter(trn_data_loader)
    val_data_iterator = iter(val_data_loader)

    for global_step in range(max_step):
        print(global_step)
        try:
            image, mask = next(trn_data_iterator)
        except StopIteration:
            trn_data_iterator = iter(trn_data_loader)
            image, mask = next(trn_data_iterator)

        image, mask = image.cuda(), mask.cuda()

        out = net(image)
        L_p = criterion(out, mask)
        loss = L_p

        optim.zero_grad()
        loss.backward()
        optim.step()

        if global_step % log_step == 0:
            loss_dict["L_p"] = round(L_p.item(), 4)
            loss_dict["L_total"] = round(loss.item(), 4)

        # Validation
        if global_step % val_step == 0:

            try:
                val_image, val_mask = next(val_data_iterator)
            except StopIteration:
                val_data_iterator = iter(val_data_loader)
                val_image, val_mask = next(val_data_iterator)

            val_image, val_mask = val_image.cuda(), val_mask.cuda()

            with torch.no_grad():
                val_out = net(val_image)
                val_L_p = criterion(val_out, val_mask)
                val_loss = val_L_p

            loss_dict["val_L_p"] = round(val_L_p.item(), 4)
            loss_dict["val_L_total"] = round(val_loss.item(), 4)
            
            print(loss_dict)

            val_out = torch.where(GaussianBlur(val_out)>0.5, 1, 0)
            trn_images = [image, mask[:, 0, :, :].unsqueeze(1), mask[:, 1, :, :].unsqueeze(1), mask[:, 2, :, :].unsqueeze(1), mask[:, 3, :, :].unsqueeze(1), out[:, 0, :, :].unsqueeze(1), out[:, 1, :, :].unsqueeze(1), out[:, 2, :, :].unsqueeze(1), out[:, 3, :, :].unsqueeze(1)]
            val_images = [val_image, val_mask[:, 0, :, :].unsqueeze(1), val_mask[:, 1, :, :].unsqueeze(1), val_mask[:, 2, :, :].unsqueeze(1), val_mask[:, 3, :, :].unsqueeze(1), val_out[:, 0, :, :].unsqueeze(1), val_out[:, 1, :, :].unsqueeze(1), val_out[:, 2, :, :].unsqueeze(1), val_out[:, 3, :, :].unsqueeze(1)]

            utils.save_image(run_desc, global_step, "trn_imgs", trn_images)
            utils.save_image(run_desc, global_step, "val_imgs", val_images)

        if global_step % ckpt_step == 0:
            utils.save_checkpoint(run_desc, net, optim, "BiSe", global_step)

if __name__ == "__main__":

    # configs
    configs = {
        "run_id": "sixth_try",
        "dataroot": "/media/deep3090/hdd/mask_dataset",
        "num_classes": 4,
        "num_workers": 8,
        "batch_size" : 16,
        "crop_size": [512, 512],
        "max_step": 80000,
        "initial_lr": 1e-2, 
        "log_step": 100,
        "val_step": 200,
        "ckpt_step": 1000,
    }

    train(configs)