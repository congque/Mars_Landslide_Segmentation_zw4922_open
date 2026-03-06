import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and (num_channels % num_groups != 0):
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
            bias=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.norm(x)
        return x, h, w


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, bias=False)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor, h: int, w: int):
        b, n, c = x.shape

        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_ = x.transpose(1, 2).reshape(b, c, h, w)
            x_ = self.sr(x_)
            x_ = x_.reshape(b, c, -1).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return x


class MixFFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, h: int, w: int):
        b, n, _ = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(b, -1, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SegFormerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim=dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim=dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, h: int, w: int):
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x


class SegFormerEncoder(nn.Module):
    def __init__(
        self,
        in_chans: int = 15,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        depths=(3, 4, 6, 3),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratios=(4.0, 4.0, 4.0, 4.0),
    ):
        super().__init__()

        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        stage_in = in_chans
        for i in range(4):
            patch_size = 7 if i == 0 else 3
            stride = 4 if i == 0 else 2
            self.patch_embeds.append(
                OverlapPatchEmbed(
                    in_chans=stage_in,
                    embed_dim=embed_dims[i],
                    patch_size=patch_size,
                    stride=stride,
                )
            )
            self.blocks.append(
                nn.ModuleList(
                    [
                        SegFormerBlock(
                            dim=embed_dims[i],
                            num_heads=num_heads[i],
                            mlp_ratio=mlp_ratios[i],
                            sr_ratio=sr_ratios[i],
                        )
                        for _ in range(depths[i])
                    ]
                )
            )
            self.norms.append(nn.LayerNorm(embed_dims[i]))
            stage_in = embed_dims[i]

    def forward(self, x: torch.Tensor):
        feats = []
        for i in range(4):
            x, h, w = self.patch_embeds[i](x)
            for blk in self.blocks[i]:
                x = blk(x, h, w)
            x = self.norms[i](x)
            x = x.transpose(1, 2).reshape(x.shape[0], -1, h, w)
            feats.append(x)
        return feats


class SegFormerDecoder(nn.Module):
    def __init__(
        self,
        embed_dims=(64, 128, 320, 512),
        decoder_dim: int = 256,
        num_classes: int = 1,
    ):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(c, decoder_dim, kernel_size=1) for c in embed_dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * len(embed_dims), decoder_dim, kernel_size=3, padding=1, bias=False),
            _group_norm(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, feats, output_size):
        target_size = feats[0].shape[2:]
        upsampled = []
        for f, proj in zip(feats, self.projs):
            x = proj(f)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            upsampled.append(x)

        x = torch.cat(upsampled, dim=1)
        x = self.fuse(x)
        x = self.cls(x)
        if x.shape[2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class SegFormer(nn.Module):
    def __init__(
        self,
        n_channels: int = 15,
        n_classes: int = 1,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        depths=(3, 4, 6, 3),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratios=(4.0, 4.0, 4.0, 4.0),
        decoder_dim: int = 256,
    ):
        super().__init__()
        self.encoder = SegFormerEncoder(
            in_chans=n_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths,
            sr_ratios=sr_ratios,
            mlp_ratios=mlp_ratios,
        )
        self.decoder = SegFormerDecoder(
            embed_dims=embed_dims,
            decoder_dim=decoder_dim,
            num_classes=n_classes,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)
        logits = self.decoder(feats, output_size=x.shape[2:])
        return logits
