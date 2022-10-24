import os 
import cv2
import torch
import torchvision


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

def save_checkpoint(run_desc, model, optimizer, name, global_step):
    
    ckpt_dict = {}
    ckpt_dict['global_step'] = global_step
    ckpt_dict['model'] = model.state_dict()
    # ckpt_dict['optimizer'] = optimizer.state_dict()

    dir_path = f'./train_result/{run_desc}/ckpt'
    os.makedirs(dir_path, exist_ok=True)
    
    ckpt_path = f'{dir_path}/{name}_{global_step}.pt'
    torch.save(ckpt_dict, ckpt_path)

    latest_ckpt_path = f'{dir_path}/{name}_latest.pt'
    torch.save(ckpt_dict, latest_ckpt_path)
        