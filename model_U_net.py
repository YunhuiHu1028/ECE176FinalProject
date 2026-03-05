import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetInpaint(nn.Module):
    def __init__(self, base_ch=64, use_mask=True):
        super().__init__()
        in_ch = 4 if use_mask else 3

        self.inc = DoubleConv(in_ch, base_ch)                       # 4, 64
        self.down1 = Down(base_ch, base_ch * 2)                     # 64, 128
        self.down2 = Down(base_ch * 2, base_ch * 4)                 # 128, 256
        self.down3 = Down(base_ch * 4, base_ch * 8)                 # 256, 512
        self.bot = Down(base_ch * 8, base_ch * 8)                   # 512, 512
        self.up1 = Up(base_ch * 8 + base_ch * 8, base_ch * 4)       # (B, 512+512, 16, 16) 
        self.up2 = Up(base_ch * 4 + base_ch * 4, base_ch * 2)       # (B, 512, 32, 32)
        self.up3 = Up(base_ch * 2 + base_ch * 2, base_ch)           # (B, 256, 64, 64)
        self.up4 = Up(base_ch + base_ch, base_ch)                   # (B, 128, 128, 128)
        self.outc = nn.Conv2d(base_ch, 3, 1)                        # (B, 3, 128, 128)
        self.act = nn.Sigmoid()

    def forward(self, x):                   # input x(B, 3, 128, 128)
        x1 = self.inc(x)                    # x1(B, 64, 128, 128)
        x2 = self.down1(x1)                 # x2(B, 128, 64, 64)
        x3 = self.down2(x2)                 # x3(B, 256, 32, 32)
        x4 = self.down3(x3)                 # x4(B, 512, 16, 16)
        xb = self.bot(x4)                   # xb(B, 512, 8, 8)
        x = self.up1(xb, x4)                # x (B, 256, 16, 16)
        x = self.up2(x,  x3)                # x (B, 128, 32, 32)
        x = self.up3(x,  x2)                # x (B, 64, 64, 64)
        x = self.up4(x,  x1)                # x (B, 64, 128, 128)
        x = self.outc(x)                    # x (B, 3, 128, 128)
        return self.act(x)