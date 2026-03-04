import os
import random
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

def random_mask_generate(H, W, min_frac=0.2, max_frac=0.5):
    m = torch.ones(1, H, W, dtype = torch.float32)
    hole_H = int(random.uniform(min_frac, max_frac) * H)
    hole_W = int(random.uniform(min_frac, max_frac) * W)
    top = random.randint(0, H - hole_H)
    width = random.randint(0, W - hole_W)
    m[: , top:top + hole_H, width:width + hole_W] = 0.0
    return m

class CeleADataset(Dataset):
    def __init__(self, img_dir, image_size=128):
        self.paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
        self.H = 128
        self.W = 128
        self.tf = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        Img = Image.open(path).convert("RGB")
        x = self.tf(Img)
        m = random_mask_generate(self.H, self.W)
        x_masked = x * m
        return x_masked, m, x