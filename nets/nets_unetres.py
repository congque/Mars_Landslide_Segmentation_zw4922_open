import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, ResNet50_Weights
try:
    from torchvision.models import (
        efficientnet_b4,
        efficientnet_b5,
        EfficientNet_B4_Weights,
        EfficientNet_B5_Weights,
    )
    _HAS_EFFICIENTNET = True
except Exception:
    efficientnet_b4 = None
    efficientnet_b5 = None
    EfficientNet_B4_Weights = None
    EfficientNet_B5_Weights = None
    _HAS_EFFICIENTNET = False
try:
    from torchvision.models import (
        convnext_base,
        convnext_small,
        ConvNeXt_Base_Weights,
        ConvNeXt_Small_Weights,
    )
    _HAS_CONVNEXT = True
except Exception:
    convnext_base = None
    convnext_small = None
    ConvNeXt_Base_Weights = None
    ConvNeXt_Small_Weights = None
    _HAS_CONVNEXT = False


def _group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and (num_channels % num_groups != 0):
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


def _resnet_norm_layer(num_channels: int) -> nn.Module:
    return _group_norm(num_channels)


def init_weights_kaiming(module: nn.Module, conv_gain: float = 1.0):
    for m in module.modules():
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

def init_unetres_weights(model: nn.Module, pretrained_backbone: bool = False, conv_gain: float = 1.0):
    if pretrained_backbone:
        init_weights_kaiming(model.encoder0[0], conv_gain=conv_gain)
        init_weights_kaiming(model.decoder4, conv_gain=conv_gain)
        init_weights_kaiming(model.decoder3, conv_gain=conv_gain)
        init_weights_kaiming(model.decoder2, conv_gain=conv_gain)
        init_weights_kaiming(model.decoder1, conv_gain=conv_gain)
        init_weights_kaiming(model.decoder0, conv_gain=conv_gain)
        init_weights_kaiming(model.final, conv_gain=conv_gain)
    else:
        init_weights_kaiming(model, conv_gain=conv_gain)

# ---------------- CBAM ----------------
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

# ---------------- Decoder Block ----------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_cbam=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(in_channels + skip_channels) if use_cbam else nn.Identity()

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.cbam(x)
        x = self.conv(x)
        return x
    
class DecoderBlock_bottle(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_cbam=True):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, kernel_size=3, padding=1),
            _group_norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            _group_norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------- U-Net with ResNet18 backbone ----------------
class UNetResNet18(nn.Module):
    def __init__(self, n_channels=15, n_classes=1, base_channel = 32, use_cbam=True, pretrained=False, conv_gain=1.0):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_cbam = use_cbam

        # ResNet18 backbone
        resnet = resnet18(pretrained=pretrained, norm_layer=_resnet_norm_layer)

        # 修改第一层卷积，支持 7 通道输入
        self.encoder0 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu
        )
        self.pool0 = resnet.maxpool  # 下采样

        self.encoder1 = resnet.layer1  # 64 channels
        self.encoder2 = resnet.layer2  # 128 channels
        self.encoder3 = resnet.layer3  # 256 channels
        self.encoder4 = resnet.layer4  # 512 channels

        # Decoder
        self.decoder4 = DecoderBlock(512, 256, 256, use_cbam)
        self.decoder3 = DecoderBlock(256, 128, 128, use_cbam)
        self.decoder2 = DecoderBlock(128, 64, 64, use_cbam)
        self.decoder1 = DecoderBlock(64, 64, 64, use_cbam)
        self.decoder0 = DecoderBlock(64, 0, 32, use_cbam)

        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

        init_unetres_weights(self, pretrained_backbone=pretrained, conv_gain=conv_gain)

    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)  # 64
        p0 = self.pool0(e0)
        e1 = self.encoder1(p0)  # 64
        e2 = self.encoder2(e1)  # 128
        e3 = self.encoder3(e2)  # 256
        e4 = self.encoder4(e3)  # 512
        # Decoder
        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)
        d0 = self.decoder0(d1)

        out = self.final(d0)
        return out

class UNetResNet34(nn.Module):
    def __init__(self, n_channels=15, n_classes=1, base_channel = 32, use_cbam=True, pretrained=False, conv_gain=1.0):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_cbam = use_cbam

        # ResNet34 backbone
        resnet = resnet34(pretrained=pretrained, norm_layer=_resnet_norm_layer)

        # 修改第一层卷积，支持 7 通道输入
        self.encoder0 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu
        )
        self.pool0 = resnet.maxpool  # 下采样

        self.encoder1 = resnet.layer1  # 64 channels
        self.encoder2 = resnet.layer2  # 128 channels
        self.encoder3 = resnet.layer3  # 256 channels
        self.encoder4 = resnet.layer4  # 512 channels

        # Decoder
        self.decoder4 = DecoderBlock(512, 256, 256, use_cbam)
        self.decoder3 = DecoderBlock(256, 128, 128, use_cbam)
        self.decoder2 = DecoderBlock(128, 64, 64, use_cbam)
        self.decoder1 = DecoderBlock(64, 64, 64, use_cbam)
        self.decoder0 = DecoderBlock(64, 0, 32, use_cbam)  # no skip

        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

        init_unetres_weights(self, pretrained_backbone=pretrained, conv_gain=conv_gain)

    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)  # 64
        p0 = self.pool0(e0)
        e1 = self.encoder1(p0)  # 64
        e2 = self.encoder2(e1)  # 128
        e3 = self.encoder3(e2)  # 256
        e4 = self.encoder4(e3)  # 512
        # Decoder
        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)
        d0 = self.decoder0(d1)

        out = self.final(d0)
        return out

class UNetResNet50(nn.Module):
    def __init__(self, n_channels=15, n_classes=1, base_channel = 32, use_cbam=True, pretrained=False, conv_gain=1.0):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_cbam = use_cbam

        # ResNet50 backbone
        resnet = resnet50(pretrained=pretrained, norm_layer=_resnet_norm_layer)

        # 修改第一层卷积，支持 7 通道输入
        self.encoder0 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu
        )
        self.pool0 = resnet.maxpool  # 下采样
        self.encoder1 = resnet.layer1  # 64 channels
        self.encoder2 = resnet.layer2  # 128 channels
        self.encoder3 = resnet.layer3  # 256 channels
        self.encoder4 = resnet.layer4  # 512 channels

        # Decoder
        self.decoder4 = DecoderBlock_bottle(2048, 1024, 512, use_cbam)
        self.decoder3 = DecoderBlock_bottle(512, 512, 256, use_cbam)
        self.decoder2 = DecoderBlock_bottle(256, 256, 128, use_cbam)
        self.decoder1 = DecoderBlock_bottle(128, 64, 64, use_cbam)
        self.decoder0 = DecoderBlock_bottle(64, 0, 32, use_cbam)


        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

        init_unetres_weights(self, pretrained_backbone=pretrained, conv_gain=conv_gain)

    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)  # 64
        p0 = self.pool0(e0)
        e1 = self.encoder1(p0)  # 64
        e2 = self.encoder2(e1)  # 128
        e3 = self.encoder3(e2)  # 256
        e4 = self.encoder4(e3)  # 512
        # Decoder
        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)
        d0 = self.decoder0(d1)

        out = self.final(d0)
        return out


def _build_convnext(variant: str, pretrained: bool):
    if not _HAS_CONVNEXT:
        raise ImportError(
            "torchvision ConvNeXt is not available in this environment. "
            "Please upgrade torchvision to use UNetConvNeXt variants."
        )
    variant = str(variant).lower()
    if variant == "base":
        if ConvNeXt_Base_Weights is not None:
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            return convnext_base(weights=weights)
        return convnext_base(pretrained=pretrained)
    if variant == "small":
        if ConvNeXt_Small_Weights is not None:
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            return convnext_small(weights=weights)
        return convnext_small(pretrained=pretrained)
    raise ValueError(f"Unsupported ConvNeXt variant: {variant}")


class UNetConvNeXt(nn.Module):
    def __init__(
        self,
        variant="base",
        n_channels=15,
        n_classes=1,
        base_channel=32,
        use_cbam=True,
        pretrained=False,
        conv_gain=1.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_cbam = use_cbam
        self.variant = str(variant).lower()
        backbone = _build_convnext(self.variant, pretrained=pretrained)
        stem = backbone.features[0][0]
        if not isinstance(stem, nn.Conv2d):
            raise RuntimeError("Unexpected ConvNeXt stem structure.")

        if n_channels != stem.in_channels:
            new_stem = nn.Conv2d(
                in_channels=n_channels,
                out_channels=stem.out_channels,
                kernel_size=stem.kernel_size,
                stride=stem.stride,
                padding=stem.padding,
                dilation=stem.dilation,
                groups=stem.groups,
                bias=(stem.bias is not None),
                padding_mode=stem.padding_mode,
            )
            if pretrained:
                with torch.no_grad():
                    new_stem.weight.zero_()
                    c_copy = min(3, n_channels)
                    new_stem.weight[:, :c_copy] = stem.weight[:, :c_copy]
                    if n_channels > c_copy:
                        mean_w = stem.weight.mean(dim=1, keepdim=True)
                        new_stem.weight[:, c_copy:] = mean_w.repeat(1, n_channels - c_copy, 1, 1)
                    if stem.bias is not None and new_stem.bias is not None:
                        new_stem.bias.copy_(stem.bias)
            backbone.features[0][0] = new_stem
            self.stem_conv = new_stem
        else:
            self.stem_conv = stem

        self.backbone = backbone

        if self.variant == "base":
            c4, c3, c2, c1 = 1024, 512, 256, 128
        elif self.variant == "small":
            c4, c3, c2, c1 = 768, 384, 192, 96
        else:
            raise ValueError(f"Unsupported ConvNeXt variant: {self.variant}")

        self.decoder4 = DecoderBlock_bottle(c4, c3, 512, use_cbam)
        self.decoder3 = DecoderBlock_bottle(512, c2, 256, use_cbam)
        self.decoder2 = DecoderBlock_bottle(256, c1, 128, use_cbam)
        self.decoder1 = DecoderBlock_bottle(128, 0, 64, use_cbam)
        self.decoder0 = DecoderBlock_bottle(64, 0, 32, use_cbam)
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

        if pretrained:
            init_weights_kaiming(self.stem_conv, conv_gain=conv_gain)
            init_weights_kaiming(self.decoder4, conv_gain=conv_gain)
            init_weights_kaiming(self.decoder3, conv_gain=conv_gain)
            init_weights_kaiming(self.decoder2, conv_gain=conv_gain)
            init_weights_kaiming(self.decoder1, conv_gain=conv_gain)
            init_weights_kaiming(self.decoder0, conv_gain=conv_gain)
            init_weights_kaiming(self.final, conv_gain=conv_gain)
        else:
            init_weights_kaiming(self, conv_gain=conv_gain)

    def forward(self, x):
        x = self.backbone.features[0](x)
        e1 = self.backbone.features[1](x)
        x = self.backbone.features[2](e1)
        e2 = self.backbone.features[3](x)
        x = self.backbone.features[4](e2)
        e3 = self.backbone.features[5](x)
        x = self.backbone.features[6](e3)
        e4 = self.backbone.features[7](x)
        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2)
        d0 = self.decoder0(d1)
        out = self.final(d0)
        return out


class UNetConvNeXtBase(UNetConvNeXt):
    def __init__(self, **kwargs):
        super().__init__(variant="base", **kwargs)


class UNetConvNeXtSmall(UNetConvNeXt):
    def __init__(self, **kwargs):
        super().__init__(variant="small", **kwargs)


def _build_efficientnet(variant: str, pretrained: bool):
    if not _HAS_EFFICIENTNET:
        raise ImportError(
            "torchvision EfficientNet is not available in this environment. "
            "Please upgrade torchvision to use UNetEfficientNetB4/B5."
        )
    variant = str(variant).lower()
    if variant == "b4":
        if EfficientNet_B4_Weights is not None:
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            return efficientnet_b4(weights=weights)
        return efficientnet_b4(pretrained=pretrained)
    if variant == "b5":
        if EfficientNet_B5_Weights is not None:
            weights = EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
            return efficientnet_b5(weights=weights)
        return efficientnet_b5(pretrained=pretrained)
    raise ValueError(f"Unsupported EfficientNet variant: {variant}")


def _collect_scale_features_efficientnet(features: nn.Sequential, x: torch.Tensor):
    h0, w0 = int(x.shape[-2]), int(x.shape[-1])
    feats = {}
    out = x
    for block in features:
        out = block(out)
        h, w = int(out.shape[-2]), int(out.shape[-1])
        if h <= 0 or w <= 0:
            continue
        sh = h0 // h
        sw = w0 // w
        if sh == sw and sh in (2, 4, 8, 16, 32):
            feats[sh] = out
    return feats


class UNetEfficientNet(nn.Module):
    def __init__(
        self,
        variant="b4",
        n_channels=15,
        n_classes=1,
        base_channel=32,
        use_cbam=True,
        pretrained=False,
        conv_gain=1.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_cbam = use_cbam
        self.variant = str(variant).lower()

        backbone = _build_efficientnet(self.variant, pretrained=pretrained)
        stem = backbone.features[0][0]
        if not isinstance(stem, nn.Conv2d):
            raise RuntimeError("Unexpected EfficientNet stem structure.")

        if n_channels != stem.in_channels:
            new_stem = nn.Conv2d(
                in_channels=n_channels,
                out_channels=stem.out_channels,
                kernel_size=stem.kernel_size,
                stride=stem.stride,
                padding=stem.padding,
                dilation=stem.dilation,
                groups=stem.groups,
                bias=(stem.bias is not None),
                padding_mode=stem.padding_mode,
            )
            if pretrained:
                with torch.no_grad():
                    new_stem.weight.zero_()
                    c_copy = min(3, n_channels)
                    new_stem.weight[:, :c_copy] = stem.weight[:, :c_copy]
                    if n_channels > c_copy:
                        mean_w = stem.weight.mean(dim=1, keepdim=True)
                        new_stem.weight[:, c_copy:] = mean_w.repeat(1, n_channels - c_copy, 1, 1)
                    if stem.bias is not None and new_stem.bias is not None:
                        new_stem.bias.copy_(stem.bias)
            backbone.features[0][0] = new_stem
            self.stem_conv = new_stem
        else:
            self.stem_conv = stem

        self.backbone = backbone

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, 128, 128, dtype=torch.float32)
            feats = _collect_scale_features_efficientnet(self.backbone.features, dummy)
        missing = [s for s in (2, 4, 8, 16, 32) if s not in feats]
        if len(missing) > 0:
            raise RuntimeError(
                f"EfficientNet-{self.variant} feature scales missing: {missing}. "
                "Input should be divisible by 32."
            )
        c2 = int(feats[2].shape[1])
        c4 = int(feats[4].shape[1])
        c8 = int(feats[8].shape[1])
        c16 = int(feats[16].shape[1])
        c32 = int(feats[32].shape[1])

        self.decoder4 = DecoderBlock(c32, c16, 256, use_cbam)
        self.decoder3 = DecoderBlock(256, c8, 128, use_cbam)
        self.decoder2 = DecoderBlock(128, c4, 96, use_cbam)
        self.decoder1 = DecoderBlock(96, c2, 64, use_cbam)
        self.decoder0 = DecoderBlock(64, 0, 32, use_cbam)
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

        if pretrained:
            init_weights_kaiming(self.stem_conv, conv_gain=conv_gain)
            init_weights_kaiming(self.decoder4, conv_gain=conv_gain)
            init_weights_kaiming(self.decoder3, conv_gain=conv_gain)
            init_weights_kaiming(self.decoder2, conv_gain=conv_gain)
            init_weights_kaiming(self.decoder1, conv_gain=conv_gain)
            init_weights_kaiming(self.decoder0, conv_gain=conv_gain)
            init_weights_kaiming(self.final, conv_gain=conv_gain)
        else:
            init_weights_kaiming(self, conv_gain=conv_gain)

    def forward(self, x):
        feats = _collect_scale_features_efficientnet(self.backbone.features, x)
        if len(feats) < 5:
            raise RuntimeError("EfficientNet encoder failed to provide full pyramid features.")

        e2 = feats[2]
        e4 = feats[4]
        e8 = feats[8]
        e16 = feats[16]
        e32 = feats[32]
        d4 = self.decoder4(e32, e16)
        d3 = self.decoder3(d4, e8)
        d2 = self.decoder2(d3, e4)
        d1 = self.decoder1(d2, e2)
        d0 = self.decoder0(d1)
        out = self.final(d0)
        return out


class UNetEfficientNetB4(UNetEfficientNet):
    def __init__(self, **kwargs):
        super().__init__(variant="b4", **kwargs)


class UNetEfficientNetB5(UNetEfficientNet):
    def __init__(self, **kwargs):
        super().__init__(variant="b5", **kwargs)
