import torch
import torch.nn as nn
import os
import random
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataset import CeleADataset
from model import InpaintNet
import matplotlib.pyplot as plt

def l1_loss(pred, target, mask, eps=1e-8):
    hole = 1.0 - mask
    hole = hole.expand_as(target)
    diff = torch.abs(pred - target) * hole
    return diff.sum() / (hole.sum() + eps)

def save_viz(x, x_masked, mask, pred, save_path, max_n=4):
    B = min(x.size(0), max_n)
    x = x[:B].detach().cpu()
    x_masked = x_masked[:B].detach().cpu()
    mask = mask[:B].detach().cpu()
    pred = pred[:B].detach().cpu()
    completed = x_masked + pred * (1.0 - mask)
    plt.figure(figsize=(12, 3 * B))
    for i in range(B):
        imgs = [x[i], x_masked[i], pred[i], completed[i]]
        titles = ["Original", "Masked", "Pred", "Completed"]
        for j in range(4):
            plt.subplot(B, 4, i * 4 + j + 1)
            plt.imshow(imgs[j].permute(1, 2, 0).clamp(0, 1))
            plt.title(titles[j])
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    img_dir = r"C:\Users\jia65\ECE176 Dataset\img_align_celeba"
    file_dir = r"C:\Users\jia65\Desktop\ECE176"
    image_size = 128
    batch_size = 32
    num_workers = 4
    lr = 1e-3
    epochs = 5
    val_frac = 0.05
    best_val_loss = float("inf")
    best_path = os.path.join(file_dir, "best_inpaint_net.pth")
    os.makedirs(file_dir, exist_ok=True)
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    data = CeleADataset(img_dir, image_size)
    n_total = len(data)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(data, [n_train, n_val], generator=generator)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False,
    )

    model  = InpaintNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for x_masked, mask, x in train_loader:
            x_masked = x_masked.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            pred = model(x_masked)
            loss = l1_loss(pred, x, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_masked, mask, x in val_loader:
                x_masked = x_masked.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                x = x.to(device, non_blocking=True)
                pred = model(x_masked)
                loss = l1_loss(pred, x, mask)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        x_masked_v, mask_v, x_v = next(iter(val_loader))
        x_masked_v = x_masked_v.to(device, non_blocking=True)
        mask_v     = mask_v.to(device, non_blocking=True)
        x_v        = x_v.to(device, non_blocking=True)

        with torch.no_grad():
            pred_v = model(x_masked_v)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, best_path)
            print(f"Epoch={epoch/epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
            viz_path = os.path.join(file_dir, "best_preview.png")
            save_viz(x_v, x_masked_v, mask_v, pred_v, viz_path)
            print(f"Saved preview: {viz_path}")
    
    print("======= Train Finish =======")

if __name__ == "__main__":
    main()