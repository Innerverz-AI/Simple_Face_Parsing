from cProfile import run
from libs.checkpoint import load_checkpoint, save_checkpoint
from libs.dataset import FaceDataset 
from libs.loss import OhemCELoss, SoftmaxFocalLoss
from libs.optimizer import Optimizer
# from models.bisenet import BiSeNet
from models.bisenet import BiSeNet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

import argparse
import os
import cv2
import sys
import wandb
from random import shuffle

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

def gray2rgb(gray_batch):
    rgb_batch = []

    for img in gray_batch:
        img = transforms.ToPILImage()(img)
        img = img.convert("RGB")
        img = transforms.ToTensor()(img)
        rgb_batch.append(img)
    
    rgb_batch = torch.stack(rgb_batch)
    return rgb_batch.cuda()

def train(baseline, cfg):
    # Dataset
    dataset = FaceDataset(cfg["data_roots"], crop_size=cfg["crop_size"], mode="train")
    trn_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - trn_size
    trn_dataset, val_dataset = random_split(dataset, [trn_size, val_size])

    # Dataloader
    trn_data_loader = DataLoader(trn_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    val_data_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
        
    # Model
    if baseline == "resnet18":
        net = BiSeNet(n_classes=cfg["num_classes"])
        ckpt = torch.load("./ckpt/faceparser.pth", map_location="cuda")
        net.load_state_dict(ckpt, strict=False) 
        # for param in net.parameters():
            # param.requires_grad = False # Y. Try both options
    else:
        raise Exception("Invalid baseline")

    net.cuda()
    net.train() 

    # Loss function
    criterion = nn.BCELoss() 

    # Optimizer
    optim = Optimizer(
        model = net,#.module,
        lr0 = cfg["initial_lr"],
        momentum = cfg["momentum"],
        wd = cfg["weight_decay"],
        warmup_steps = cfg["warmup_step"],
        warmup_start_lr = cfg["warmup_start_lr"],
        max_iter = cfg["max_step"],
        power = cfg["power"]
    ) # Y. custom optimizer for learning rate scheduling

    # Train loop
    run_desc = cfg["baseline"]+"/"+cfg["run_id"]
    batch_size = cfg["batch_size"]
    max_step = cfg["max_step"]
    log_step = cfg["log_step"]
    val_step = cfg["val_step"]
    ckpt_step = cfg["ckpt_step"]

    trn_data_iterator = iter(trn_data_loader)

    for global_step in range(max_step):
        try:
            image, mask = next(trn_data_iterator)
        except StopIteration:
            trn_data_iterator = iter(trn_data_loader)
            image, mask = next(trn_data_iterator)

        image = image.cuda()
        mask = mask.cuda()
        # H, W = image.size()[2:]
        mask = torch.squeeze(mask, 1)

        optim.zero_grad()
        out, out16, out32 = net(image)
        L_p = criterion(out, mask)      # Y. Principal loss; Binary classification losses shoule be considered
        L_a1 = criterion(out16, mask)   # Y. Auxiliary losses
        L_a2 = criterion (out32, mask)
        loss = L_p + L_a1 + L_a2
        loss.backward()
        optim.step()

        if global_step % log_step == 0:
            loss_dict = {}
            loss_dict["L_p"] = round(L_p.item(), 4)
            loss_dict["L_a1"] = round(L_a1.item(), 4)
            loss_dict["L_a2"] = round(L_a2.item(), 4) 
            loss_dict["L_total"] = round(loss.item(), 4)

            # Validation
            if global_step % val_step == 0:
                val_image, val_mask = next(iter(val_data_loader))
                val_image, val_mask = val_image.cuda(), val_mask.cuda()
                val_mask = torch.squeeze(val_mask, 1)

                with torch.no_grad():
                    val_out, val_out16, val_out32 = net(val_image)
                    val_L_p = criterion(val_out, val_mask) 
                    val_L_a1 = criterion(val_out16, val_mask)
                    val_L_a2 = criterion(val_out32, val_mask)
                    val_loss = val_L_p + val_L_a1 + val_L_a2

                loss_dict["val_L_p"] = round(val_L_p.item(), 4)
                loss_dict["val_L_a1"] = round(val_L_a1.item(), 4)
                loss_dict["val_L_a2"] = round(val_L_a2.item(), 4) 
                loss_dict["val_L_total"] = round(val_loss.item(), 4)

                trn_images = [image, gray2rgb(mask), gray2rgb(out)]
                val_images = [val_image, gray2rgb(val_mask), gray2rgb(val_out)]
                save_image(run_desc, global_step, "trn_imgs", trn_images)
                save_image(run_desc, global_step, "val_imgs", val_images)

            wandb.log(loss_dict)

        if global_step % ckpt_step == 0:
            save_checkpoint(run_desc, net, optim, "BiSe", global_step)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument()
    # args = parser.parse_args()

    torch.manual_seed(1012)

    # wandb init
    baseline = "resnet18" # sys.argv[1]
    run_id = sys.argv[1]
    wandb.init(project="BiSeNet", name=f"{baseline}-{run_id}")

    # configs
    configs = {
        "baseline": baseline,
        "run_id": run_id,
        "num_classes": 19,
        "num_workers": 8,
        "batch_size" : 16,
        "crop_size": [448, 448],
        "data_roots": [
            "/home/compu/dataset/CelebHQ",
            "/home/leee/data/CelebHQ-mask"
        ],
         
        # Optimzier related
        "max_step": 80000,
        "initial_lr": 1e-2, 
        "warmup_start_lr": 1e-5, 
        "warmup_step": 1000,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "power": 0.9,
        "log_step": 100,
        "val_step": 200,
        "ckpt_step": 10000,
    }

    train(baseline, configs)