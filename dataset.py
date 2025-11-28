
# 导入相关包
import torchvision
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from model import *
from mnist_data import *

class image_data_set(Dataset):
    def __init__(self, data):
        self.images = data[:,:,:,None]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(64, interpolation=InterpolationMode.BICUBIC),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx])