import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and (num_channels % num_groups != 0):
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


def init_weights_strong(model: nn.Module, conv_gain: float = 1.15):
    """
    A stronger initialization than default:
    - Conv/ConvTranspose: Kaiming normal (ReLU) + optional gain boost
    - GroupNorm: weight=1, bias=0
    - Linear: Kaiming normal, bias=0
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if conv_gain != 1.0:
                with torch.no_grad():
                    m.weight.mul_(conv_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# conv blocks
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, use_cbam=True):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            _group_norm(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_channels) if use_cbam else nn.Identity()

    def forward(self, x):
        return self.cbam(self.double_conv(x))

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class MultiScaleDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        mid_channels = out_channels // 2

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(mid_channels),
            nn.ReLU(inplace=True),
            
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=5, padding=2, bias=False),
            _group_norm(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_7 = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=7, padding=3, bias=False),
            _group_norm(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 3, out_channels, kernel_size=1, bias=False),
            _group_norm(out_channels),
        )

        # residual
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            _group_norm(out_channels)
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.init_conv(x)

        b3 = self.conv_3(x0)
        b5 = self.conv_5(x0)
        b7 = self.conv_7(x0)

        out = torch.cat([b3, b5, b7], dim=1)
        out = self.fuse(out)

        res = self.residual(x)

        return self.act(out + res)

class MultiScaleDoubleConv_Heavy(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        mid_channels = out_channels // 2

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(mid_channels),
            nn.ReLU(inplace=True),
            
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, bias=False),
            _group_norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2, bias=False),
            _group_norm(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_7 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=3, bias=False),
            _group_norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=7, padding=3, bias=False),
            _group_norm(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 3, out_channels, kernel_size=1, bias=False),
            _group_norm(out_channels),
        )

        # residual
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            _group_norm(out_channels)
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        b3 = self.conv_3(x)
        b5 = self.conv_5(x)
        b7 = self.conv_7(x)

        out = torch.cat([b3, b5, b7], dim=1)
        out = self.fuse(out)

        res = self.residual(x)

        return self.act(out + res)

class MultiScaleDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            MultiScaleDoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class MultiScaleDown_Heavy(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            MultiScaleDoubleConv_Heavy(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class MultiScaleUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = MultiScaleDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class MultiScaleUp_Heavy(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = MultiScaleDoubleConv_Heavy(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# main nets
class UNet(nn.Module):

    def __init__(self, n_channels=13, n_classes=1, base_channel=32):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, base_channel)
        self.down1 = Down(base_channel, base_channel*2)
        self.down2 = Down(base_channel*2, base_channel*4)
        self.down3 = Down(base_channel*4, base_channel*8)
        self.down4 = Down(base_channel*8, base_channel*16)

        self.up1 = Up(base_channel*16 + base_channel*8, base_channel*8)
        self.up2 = Up(base_channel*8 + base_channel*4, base_channel*4)
        self.up3 = Up(base_channel*4 + base_channel*2, base_channel*2)
        self.up4 = Up(base_channel*2 + base_channel, base_channel)

        self.outc = OutConv(base_channel, n_classes)

        init_weights_strong(self, conv_gain=1.0)

    def forward(self, x):
        
        x1 = self.inc(x)  # 128×128, base_channel
        x2 = self.down1(x1)  # 64×64, base_channel*2
        x3 = self.down2(x2)  # 32×32, base_channel*4
        x4 = self.down3(x3)  # 16×16, base_channel*8
        x5 = self.down4(x4)  # 8×8, base_channel*16
        x = self.up1(x5, x4)  # 16×16, base_channel*8
        x = self.up2(x, x3)  # 32×32, base_channel*4
        x = self.up3(x, x2)  # 64×64, base_channel*2
        x = self.up4(x, x1)  # 128×128, base_channel

        logits = self.outc(x)  # 128×128, 1

        return logits

class UNet_MS(nn.Module):

    def __init__(self, n_channels=13, n_classes=1, base_channel=32):
        super(UNet_MS, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_channel = base_channel

        self.inc = MultiScaleDoubleConv(n_channels, base_channel)
        self.down1 = MultiScaleDown(base_channel, base_channel*2)
        self.down2 = MultiScaleDown(base_channel*2, base_channel*4)
        self.down3 = MultiScaleDown(base_channel*4, base_channel*8)
        self.down4 = MultiScaleDown(base_channel*8, base_channel*16)
        self.up1 = MultiScaleUp(base_channel*16 + base_channel*8, base_channel*8)
        self.up2 = MultiScaleUp(base_channel*8 + base_channel*4, base_channel*4)
        self.up3 = MultiScaleUp(base_channel*4 + base_channel*2, base_channel*2)
        self.up4 = MultiScaleUp(base_channel*2 + base_channel, base_channel)

        self.outc = OutConv(base_channel, n_classes)

        init_weights_strong(self, conv_gain=1.0)

    def forward(self, x):
        
        x1 = self.inc(x)  # 128×128, base_channel
        x2 = self.down1(x1)  # 64×64, base_channel*2
        x3 = self.down2(x2)  # 32×32, base_channel*4
        x4 = self.down3(x3)  # 16×16, base_channel*8
        x5 = self.down4(x4)  # 8×8, base_channel*16
        x = self.up1(x5, x4)  # 16×16, base_channel*8
        x = self.up2(x, x3)  # 32×32, base_channel*4
        x = self.up3(x, x2)  # 64×64, base_channel*2
        x = self.up4(x, x1)  # 128×128, base_channel

        logits = self.outc(x)  # 128×128, 1

        return logits

class UNet_Half_Down(nn.Module):

    def __init__(self, n_channels=13, n_classes=1, base_channel=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, base_channel)
        self.down1 = Down(base_channel, base_channel*2)
        self.down2 = Down(base_channel*2, base_channel*4)
        self.down3 = DoubleConv(base_channel*4, base_channel*8)
        self.down4 = DoubleConv(base_channel*8, base_channel*16)

        self.up1 = Up(base_channel*16 + base_channel*8, base_channel*8)
        self.up2 = Up(base_channel*8 + base_channel*4, base_channel*4)
        self.up3 = Up(base_channel*4 + base_channel*2, base_channel*2)
        self.up4 = Up(base_channel*2 + base_channel, base_channel)

        self.outc = OutConv(base_channel, n_classes)

        init_weights_strong(self, conv_gain=1.0)

    def forward(self, x):
        
        x1 = self.inc(x)  # 128×128, base_channel
        x2 = self.down1(x1)  # 64×64, base_channel*2
        x3 = self.down2(x2)  # 32×32, base_channel*4
        x4 = self.down3(x3)  # 16×16, base_channel*8
        x5 = self.down4(x4)  # 8×8, base_channel*16

        x = self.up1(x5, x4)  # 16×16, base_channel*8
        x = self.up2(x, x3)  # 32×32, base_channel*4
        x = self.up3(x, x2)  # 64×64, base_channel*2
        x = self.up4(x, x1)  # 128×128, base_channel

        logits = self.outc(x)  # 128×128, 1

        return logits

class UNet_MSHD_Heavy(nn.Module):

    def __init__(self, n_channels=15, n_classes=1, base_channel=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_channel = base_channel

        self.inc = MultiScaleDoubleConv(n_channels, base_channel)
        self.down1 = MultiScaleDown_Heavy(base_channel, base_channel*2)
        self.down2 = MultiScaleDown_Heavy(base_channel*2, base_channel*4)
        self.down3 = MultiScaleDown_Heavy(base_channel*4, base_channel*8)
        self.down4 = MultiScaleDoubleConv_Heavy(base_channel*8, base_channel*16)
        self.up1 = MultiScaleUp_Heavy(base_channel*16 + base_channel*8, base_channel*8)
        self.up2 = MultiScaleUp_Heavy(base_channel*8 + base_channel*4, base_channel*4)
        self.up3 = MultiScaleUp_Heavy(base_channel*4 + base_channel*2, base_channel*2)
        self.up4 = MultiScaleUp_Heavy(base_channel*2 + base_channel, base_channel)

        self.outc = OutConv(base_channel, n_classes)

        init_weights_strong(self, conv_gain=1.0)

    def forward(self, x):
        
        x1 = self.inc(x)  # 128×128, base_channel
        x2 = self.down1(x1)  # 64×64, base_channel*2
        x3 = self.down2(x2)  # 32×32, base_channel*4
        x4 = self.down3(x3)  # 16×16, base_channel*8
        x5 = self.down4(x4)  # 8×8, base_channel*16
        x = self.up1(x5, x4)  # 16×16, base_channel*8
        x = self.up2(x, x3)  # 32×32, base_channel*4
        x = self.up3(x, x2)  # 64×64, base_channel*2
        x = self.up4(x, x1)  # 128×128, base_channel

        logits = self.outc(x)  # 128×128, 1

        return logits

class UNet_MSHD_Light(nn.Module):

    def __init__(self, n_channels=13, n_classes=1, base_channel=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_channel = base_channel

        self.inc = MultiScaleDoubleConv(n_channels, base_channel)
        self.down1 = MultiScaleDown(base_channel, base_channel*2)
        self.down2 = MultiScaleDown(base_channel*2, base_channel*4)
        self.down3 = MultiScaleDoubleConv(base_channel*4, base_channel*8)
        self.down4 = MultiScaleDoubleConv(base_channel*8, base_channel*16)
        self.up1 = MultiScaleUp(base_channel*16 + base_channel*8, base_channel*8)
        self.up2 = MultiScaleUp(base_channel*8 + base_channel*4, base_channel*4)
        self.up3 = MultiScaleUp(base_channel*4 + base_channel*2, base_channel*2)
        self.up4 = MultiScaleUp(base_channel*2 + base_channel, base_channel)

        self.outc = OutConv(base_channel, n_classes)

        init_weights_strong(self, conv_gain=1.0)

    def forward(self, x):
        
        x1 = self.inc(x)  # 128×128, base_channel
        x2 = self.down1(x1)  # 64×64, base_channel*2
        x3 = self.down2(x2)  # 32×32, base_channel*4
        x4 = self.down3(x3)  # 16×16, base_channel*8
        x5 = self.down4(x4)  # 8×8, base_channel*16
        
        x = self.up1(x5, x4)  # 16×16, base_channel*8
        x = self.up2(x, x3)  # 32×32, base_channel*4
        x = self.up3(x, x2)  # 64×64, base_channel*2
        x = self.up4(x, x1)  # 128×128, base_channel

        logits = self.outc(x)  # 128×128, 1

        return logits

if __name__ == "__main__":
    model = UNet(n_channels=7, n_classes=1)
    test_input = torch.randn(2, 7, 128, 128)
    output = model(test_input)
    print(f"输入尺寸: {test_input.shape}")
    print(f"输出尺寸: {output.shape}")
    print(f"输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
