import os
from torchvision import  datasets
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
path = "C:/Users/Avon/Desktop/dataset"
image_path = os.listdir(path+'/img')
mask_path = os.listdir(path+'/mask')
class groupDataset(Dataset):
    def __init__(self,image_path,mask_path):
        self.image_path=image_path
        self.mask_path=mask_path
    def __getitem__(self, item):
        img = Image.open(path+'/img/'+self.image_path[item]).convert('RGB')
        msk = Image.open(path+'/mask/'+self.mask_path[item]).convert('1')
        img_tensor = transforms.ToTensor()(img)
        msk_tensor = transforms.ToTensor()(msk)
        img_tensor = transforms.Resize([800,400])(img_tensor)
        msk_tensor = transforms.Resize([800,400])(msk_tensor)
        return img_tensor,msk_tensor
    def __len__(self):
        return len(self.image_path)
Dataset = groupDataset(image_path,mask_path)
train_dataset,test_dataset = torch.utils.data.random_split(Dataset,[0.8,0.2])
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)