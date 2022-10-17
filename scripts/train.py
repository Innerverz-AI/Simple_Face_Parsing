from libs.checkpoint import load_checkpoint, save_checkpoint
from libs.dataset import FaceDataset 
from models.bisenet import BiSeNet

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split

import os
import cv2
import sys
import wandb

def save_image(run_desc, global_step, dir, images):
    dir_path = f'train_result/{run_desc}/{dir}'
    os.makedirs(dir_path, exist_ok=True)
    
    sample_image = make_grid_image(images).detach().cpu().numpy().transpose([1,2,0]) * 255
    cv2.imwrite(f'{dir_path}/{str(global_step).zfill(8)}.jpg', sample_image[:,:,::-1])

def make_grid_image(images_list):
    grid_rows = []

    for i, images in enumerate(images_list):
        images = images[:8] # Drop images if there are more than 8 images in the list
        grid_row = torchvision.utils.make_grid(images, nrow=images.shape[0]) 
        if i == 0: # image
            grid_row = grid_row * 0.5 + 0.5
        grid_rows.append(grid_row)

    grid = torch.cat(grid_rows, dim=1)
    return grid

def train(cfg):
    # Dataset
    dataset = FaceDataset()
    trn_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - trn_size
    trn_dataset, val_dataset = random_split(dataset, [trn_size, val_size])

    # Dataloader
    trn_data_loader = DataLoader(trn_dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_data_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
    
    # Model
    net = BiSeNet(n_classes=cfg["num_classes"]).cuda().train()
    ckpt = torch.load("./face_parsing/ckpt/faceparser.pth", map_location="cuda")
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

    for global_step in range(max_step):
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
            print(global_step)
            val_image, val_mask = next(iter(val_data_loader))
            val_image, val_mask = val_image.cuda(), val_mask.cuda()

            with torch.no_grad():
                val_out = net(val_image)
                val_L_p = criterion(val_out, val_mask)
                val_loss = val_L_p

            loss_dict["val_L_p"] = round(val_L_p.item(), 4)
            loss_dict["val_L_total"] = round(val_loss.item(), 4)

            trn_images = [image, mask[:, 0, :, :].unsqueeze(1), mask[:, 1, :, :].unsqueeze(1), mask[:, 2, :, :].unsqueeze(1), mask[:, 3, :, :].unsqueeze(1), out[:, 0, :, :].unsqueeze(1), out[:, 1, :, :].unsqueeze(1), out[:, 2, :, :].unsqueeze(1), out[:, 3, :, :].unsqueeze(1)]
            val_images = [val_image, val_mask[:, 0, :, :].unsqueeze(1), val_mask[:, 1, :, :].unsqueeze(1), val_mask[:, 2, :, :].unsqueeze(1), val_mask[:, 3, :, :].unsqueeze(1), val_out[:, 0, :, :].unsqueeze(1), val_out[:, 1, :, :].unsqueeze(1), val_out[:, 2, :, :].unsqueeze(1), val_out[:, 3, :, :].unsqueeze(1)]

            save_image(run_desc, global_step, "trn_imgs", trn_images)
            save_image(run_desc, global_step, "val_imgs", val_images)

            wandb.log(loss_dict)

        if global_step % ckpt_step == 0:
            save_checkpoint(run_desc, net, optim, "BiSe", global_step)

if __name__ == "__main__":

    torch.manual_seed(1012)

    # wandb init
    run_id = sys.argv[1]
    wandb.init(project="BiSeNet", name=run_id)

    # configs
    configs = {
        "run_id": run_id,
        "num_classes": 4,
        "num_workers": 8,
        "batch_size" : 16,
        "crop_size": [512, 512],
        "max_step": 80000,
        "initial_lr": 1e-2, 
        "log_step": 100,
        "val_step": 200,
        "ckpt_step": 10000,
    }

    train(configs)