import torch
import os

        
def load_checkpoint(ckpt_desc, ckpt_step, model, optimizer, name):
    ckpt_step = "latest" if ckpt_step is None else ckpt_step
    ckpt_path = f'./train_result/{ckpt_desc}/ckpt/{name}_{ckpt_step}.pt'
    
    ckpt_dict = torch.load(ckpt_path, map_location=torch.device('cuda'))
    model.load_state_dict(ckpt_dict['model'], strict=False)
    # optimizer.load_state_dict(ckpt_dict['optimizer'])

    return ckpt_dict['global_step']

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
        