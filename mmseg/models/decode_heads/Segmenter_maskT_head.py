import torch.nn as nn

from ..builder import HEADS
from .decode_head import BaseDecodeHead

import torch
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn.bricks.drop import DropPath
from mmcv.cnn import ConvModule
from .Segmenter_linear_head import init_weights, trunc_normal_


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SFP(nn.Module):
    '''multi_scale_adapter SFP'''
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factors,
                 num_outs,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(SFP, self).__init__()
        self.out_channels = out_channels
        self.scale_factors = scale_factors
        self.num_ins = len(scale_factors)
        self.num_outs = num_outs
        self.stages = []
        for idx, scale in enumerate(scale_factors):
            out_dim = dim = in_channels[idx]
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, 2, stride=2, padding=0),
                    nn.GroupNorm(1, dim // 2, eps=1e-6),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        dim // 2, dim // 4, 2, stride=2, padding=0)
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, 2, stride=2, padding=0)
                ]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
            else:
                raise NotImplementedError(
                    f'scale_factor={scale} is not supported yet.')

            layers.extend([
                ConvModule(
                    out_dim,
                    out_channels[idx],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False),
                ConvModule(
                    out_channels[idx],
                    out_channels[idx],
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
            ])

            layers = nn.Sequential(*layers)
            self.add_module(f'sfp_{idx}', layers)
            self.stages.append(layers)

    def init_weights(self):
        pass

    def forward(self, inputs):
        """Forward function."""
        # print(inputs.shape)
        features = inputs
        outs = []

        # part 1: build simple feature pyramid
        for stage in self.stages:
            outs.append(stage(features))

        # part 2: add extra levels
        if self.num_outs > self.num_ins:
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)


@HEADS.register_module()
class SegmenterHead_maskT(BaseDecodeHead):
    def __init__(self, img_size_ori, n_layers, n_heads, d_model, d_ff, drop_path_rate, dropout, multi_scale_adapter, **kwargs):
        super(SegmenterHead_maskT, self).__init__(**kwargs)
        self.img_size_ori = img_size_ori
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.multi_scale_adapter = multi_scale_adapter
        if multi_scale_adapter:
            self.SFP = SFP(
                in_channels=(1024, 1024, 1024, 1024),
                out_channels=(256, 512, 1024, 1024),
                scale_factors=(4.0, 2.0, 1.0, 0.5),
                num_outs=4,
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, self.num_classes, d_model))
        
        if self.multi_scale_adapter:
            self.proj_dec = nn.Linear(2816, d_model)
        elif self.input_transform:
            self.proj_dec = nn.Linear(4096, d_model)
        else:
            self.proj_dec = nn.Linear(1024, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, inputs):
        # TODO
        features, latent = inputs
        if self.multi_scale_adapter:
            assert self.input_transform == "resize_concat"
            x = self._transform_inputs(self.SFP(latent))
        elif self.input_transform:
            x = self._transform_inputs(features)
        else:
            x = latent
        
        GS = x.shape[-1]
        x = rearrange(x, "b n h w -> b (h w) n")
        x = self.proj_dec(x)
        
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.num_classes], x[:, -self.num_classes:]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
        masks = F.interpolate(masks, size=(self.img_size_ori, self.img_size_ori), mode="bilinear")
        
        # print(f'output: {masks.shape}')
        return masks

