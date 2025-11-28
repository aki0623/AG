# 导入相关包
import torchvision
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from model import *
from mnist_data import *
from model import Generator, Discriminator
from AnoGAN_Trainer import anogan_trainer
from utils import parse_args

def main(args):
    trainer = anogan_trainer(args) 
    trainer.train()
    trainer.test()



if __name__ == '__main__':
    args = parse_args()
    main(args)
