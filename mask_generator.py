import os
import torch
import random
from glob import glob
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

img_dir = r"C:\Users\jia65\ECE176 Dataset\img_align_celeba"
paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
assert len(paths) > 0, "Fold is empyt"

path = random.choice(paths)
img = Image.open(path).convert("RGB")

tf = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
x = tf(img) #shape (3, 128, 128)

H, W = 128, 128
m = torch.ones(1, H, W)
min_frac, max_frac = 0.2, 0.5
hole_h = int(random.uniform(min_frac, max_frac) * H)
hole_w = int(random.uniform(min_frac, max_frac) * W)
top = random.randint(0, H - hole_h)
width = random.randint(0, W - hole_w)
m[:, top:top + hole_h, width:width + hole_w] = 0.0
x_masked= x * m

print("x       :", x.shape, x.min().item(), x.max().item())
print("m       :", m.shape, m.min().item(), m.max().item(), "hole area =", (m == 0).float().mean().item())
print("x_masked:", x_masked.shape, x_masked.min().item(), x_masked.max().item())
print("sample file:", path)

x_np = x.permute(1,2,0).numpy()
x_masked_np = x_masked.permute(1,2,0).numpy()
m_np = m.squeeze(0).numpy()

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.imshow(x_np)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(x_masked_np)
plt.title("Masked")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(m_np, cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.show()