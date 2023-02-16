import torch
from einops import rearrange
from torch import nn
from modules import DoubleConv
from modules import Up as upsampling
# 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
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x

class OverlapPatchMerging(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=overlap_size,
                padding=patch_size // 2,
                bias=False
            ),
            LayerNorm2d(out_channels)
        )

class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio, device = 'cuda'
            ),
            LayerNorm2d(channels, device = 'cuda'),
        )
        self.att = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True, device = 'cuda'
        )

    def forward(self, x):
        _, _, h, w = x.shape
        reduced_x = self.reducer(x)
        # attention needs tensor of shape (batch, sequence_length, channels)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")
        x = rearrange(x, "b c h w -> b (h w) c")
        out = self.att(x, reduced_x, reduced_x)[0]
        # reshape it back to (batch, channels, height, width)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out
    
class MixMLP(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            # dense layer
            nn.Conv2d(channels, channels, kernel_size=1, device = 'cuda'),
            # depth wise conv
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                groups=channels,
                padding=1,
                device = 'cuda'
            ),
            nn.GELU(),
            # dense layer
            nn.Conv2d(channels * expansion, channels, kernel_size=1, device = 'cuda'),
        )
from torchvision.ops import StochasticDepth

class ResidualAdd(nn.Module):
    """Just an util layer"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x

class SegFormerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        drop_path_prob: float = .0,
        device: str = 'cuda',
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels, device = device),
                    EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels, device = device),
                    MixMLP(channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch")
                )
            ),
        )

from typing import List

class SegFormerEncoderStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        drop_probs: List[int],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        device = "cuda",
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels, out_channels, patch_size, overlap_size,
        )
        self.blocks = nn.Sequential(
            *[
                SegFormerEncoderBlock(
                    out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i], device
                )
                for i in range(depth)
            ],
        )
        self.norm = LayerNorm2d(out_channels)

from typing import Iterable

def chunks(data: Iterable, sizes: List[int]):
    """
    Given an iterable, returns slices using sizes as indices
    """
    curr = 0
    for size in sizes:
        chunk = data[curr: curr + size]
        curr += size
        yield chunk

def time_emb_layer(x, t, emb_dim = 256):
    out_channels = x.shape[-3]
    emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels, device = 'cuda'))
    emb = emb_layer(t.to('cuda'))[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]).to('cuda')
    return emb

class SegFormerEncoder(nn.Module):
    def __init__(self, args):
        encoder_depth = args.encoder_depth
        in_channels = args.in_channels
        widths = args.widths
        depths = args.depths
        all_num_heads = args.all_num_heads
        patch_sizes = args.patch_sizes
        overlap_sizes = args.overlap_sizes
        reduction_ratios = args.reduction_ratios
        mlp_expansions = args.mlp_expansions
        drop_prob = args.drop_prob
        self.device = args.device
        super().__init__()
        # create drop paths probabilities (one for each stage's block)
        drop_probs =  [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args_encoder)
                for args_encoder in zip(
                    [in_channels, *widths],
                    widths,
                    patch_sizes,
                    overlap_sizes,
                    chunks(drop_probs, sizes=depths),
                    depths,
                    reduction_ratios,
                    all_num_heads,
                    mlp_expansions,
                    [self.device]*encoder_depth,
                )
            ]
        )
        
    def forward(self, x, t):
        features = []
        for stage in self.stages:
            x = stage(x).to(self.device)# device = self.device)#.to(self.device)
            emb = time_emb_layer(x, t)
            features.append(x + emb)
        return features

class SegFormerDecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2, device  = 'cuda'):
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, device = device),
        )

class SegFormerDecoder(nn.Module):
    def __init__(self, args):
        out_channels = args.decoder_channels
        widths = args.widths[::-1]
        scale_factors = args.scale_factors
        self.device = args.device
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(in_channels, out_channels, scale_factor, self.device)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )
    
    def forward(self, features, t):
        new_features = []
        for feature, stage in zip(features, self.stages):
            x = stage(feature).to(self.device)
            emb = time_emb_layer(x, t)
            new_features.append(x + emb)
        return new_features


class SegFormerSegmentationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(), # why relu? Who knows
            nn.BatchNorm2d(channels) # why batchnorm and not layer norm? Idk
        )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x

class SegFormer(nn.Module):
    def __init__(self, args):
        self.args = args
        super().__init__()
        self.encoder = SegFormerEncoder(self.args)
        self.decoder = SegFormerDecoder(self.args)
        self.head = SegFormerSegmentationHead(args.decoder_channels, args.num_classes,\
                                            num_features=len(args.widths))

    def pos_encoding(self, t, channels):
        t = t.unsqueeze(-1).type(torch.float)
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.args.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.to(self.args.device)

    def perform_upsampling(self, feature_map, skip_connection = None):
        # perform upsampling on the output feature map
        height, width = feature_map.shape[-2], feature_map.shape[-1]
        input_channels = feature_map.shape[-3]
        ouput_chanenls = input_channels//2
        upsample = upsampling(input_channels, ouput_chanenls)
        feature_map = upsample(feature_map, skip_connection, self.t)
        return feature_map

    def forward(self, x, t):
        self.t = self.pos_encoding(t, self.args.time_dim) # [batch_size * 256]

        # Encoder
        features = self.encoder(x, self.t)

        # Decoder
        features = self.decoder(features[::-1], self.t) # pass in reverse order

        # Reduce the channel dimention
        for i in range(len(features)):
            while(features[i].shape[-1] is not x.shape[-1]):
                features[i] = self.perform_upsampling(features[i])
                time_emb = time_emb_layer(features[i], self.t)
                features[i] = features[i] + time_emb
            if(features[i].shape[-1] == x.shape[-1]):
                # reduce the no of channels
                in_channels, out_channels = x.shape[-1], x.shape[-1]//4
                conv = nn.Sequential(
                    DoubleConv(in_channels, in_channels, residual=True),
                    DoubleConv(in_channels, out_channels, in_channels // 2),
                )
                features[i] = conv(features[i])
                time_emb = time_emb_layer(features[i], self.t)
                features[i] = features[i] + time_emb


        if(self.args.segmentation):
            # perfrom segmentation, this itself will stack all feature maps to tensor object
            features = self.head(features) # shape of (H/4, W/4)
        else:
            # stack all feature map to tensor object
            temp = features[1:]
            features = torch.cat(temp, dim=1)
        #print(features.shape)
        # make N feature maps to 3 channels
        org_height = features.shape[1]
        output = nn.Conv2d(org_height, self.args.out_channels, kernel_size=1, device = self.args.device)
        segmentation = output(features)
        #print('segmentation', segmentation.shape)
        return segmentation