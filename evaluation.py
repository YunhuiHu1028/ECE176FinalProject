import torch
from torch.utils.data import DataLoader, random_split
from dataset import CeleADataset
from model_U_net import UNetInpaint


def split_dataset(data, val_frac=0.05, test_frac=0.05, seed=42):
    n_total = len(data)
    n_val = int(n_total * val_frac)
    n_test = int(n_total * test_frac)
    n_train = n_total - n_val - n_test
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(data, [n_train, n_val, n_test], generator=generator)
    return train_ds, val_ds, test_ds


def l1_loss(pred, target, mask, eps=1e-8):
    hole = 1.0 - mask
    hole = hole.expand_as(target)
    diff = torch.abs(pred - target) * hole
    return diff.sum() / (hole.sum() + eps)


def main():
    img_dir = r"C:\Users\jia65\ECE176 Dataset\img_align_celeba"
    ckpt_path = r"C:\Users\jia65\Desktop\ECE176\checkpoints\best_inpaint_net_v2.pth"
    image_size = 128
    batch_size = 32
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    dataset = CeleADataset(img_dir, image_size)
    _, _, test_ds = split_dataset(dataset, seed=seed)
    print("Test dataset size:", len(test_ds))
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    model = UNetInpaint(base_ch=64, use_mask=True).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_masked, mask, x in test_loader:
            x_masked = x_masked.to(device)
            mask = mask.to(device)
            x = x.to(device)
            inp = torch.cat([x_masked, mask], dim=1)
            pred = model(inp)
            completed = x_masked * mask + pred * (1.0 - mask)
            loss = l1_loss(completed, x, mask)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print("Test L1 Error:", avg_loss)

if __name__ == "__main__":
    main()