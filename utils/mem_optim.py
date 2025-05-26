from tqdm import tqdm
import torch

def preload_data(dataloader, desc = 'dataloader', x_type=torch.uint8, y_type=torch.uint16):
    data = []

    for vid, y_label, y_target in tqdm(dataloader, desc):
        data.append((vid.to(x_type),y_label.to(y_type),y_target.to(y_type)))
    
    return data