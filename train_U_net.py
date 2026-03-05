import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import CeleADataset
from model_U_net import UNetInpaint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    completed = x_masked * mask + pred * (1.0 - mask)

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
    seed = 42
    set_seed(seed)
    ckpt_dir = os.path.join(file_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_inpaint_net_v2.pth")
    viz_path = os.path.join(ckpt_dir, "best_preview_v2.png")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
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
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )
    model = UNetInpaint(base_ch=64, use_mask=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_masked, mask, x in train_loader:
            x_masked = x_masked.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            inp = torch.cat([x_masked, mask], dim=1)
            pred = model(inp)
            completed = x_masked * mask + pred * (1.0 - mask)
            loss_hole = l1_loss(completed, x, mask)
            loss_valid = torch.mean(torch.abs(completed - x) * mask)
            loss = loss_hole * 6.0 + loss_valid
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))
        model.eval()
        val_loss = 0.0

        first_batch = None
        with torch.no_grad():
            for x_masked, mask, x in val_loader:
                if first_batch is None:
                    first_batch = (x_masked, mask, x)
                x_masked = x_masked.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                x = x.to(device, non_blocking=True)
                inp = torch.cat([x_masked, mask], dim=1)
                pred = model(inp)
                completed = x_masked * mask + pred * (1.0 - mask)
                loss = l1_loss(completed, x, mask)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        print(f"Epoch={epoch}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "mask_definition": "mask=1 known, 0 hole",
                    "image_size": image_size,
                },
                best_path,
            )
            print(f"Saved BEST: {best_path} (val_loss={best_val_loss:.6f})")

            x_masked_v, mask_v, x_v = first_batch
            x_masked_v = x_masked_v.to(device, non_blocking=True)
            mask_v = mask_v.to(device, non_blocking=True)
            x_v = x_v.to(device, non_blocking=True)

            with torch.no_grad():
                pred_v = model(torch.cat([x_masked_v, mask_v], dim=1))

            save_viz(x_v, x_masked_v, mask_v, pred_v, viz_path)
            print(f"Saved preview: {viz_path}")

    print("======= Train Finish =======")


if __name__ == "__main__":
    main()