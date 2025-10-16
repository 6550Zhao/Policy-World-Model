"""
Modified from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py#L34
"""

import math
from typing import Tuple, Union
from diffusers.models.activations import get_activation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class DepthToSpaceUpsample(nn.Module):
    def __init__(
        self,
        in_channels,
    ):
        super().__init__()
        conv = nn.Conv2d(in_channels, in_channels * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange("b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2),
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 4) ...")

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        out = self.net(x)
        return out


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def unpack_time(t, batch):
    _, c, w, h = t.size()
    out = torch.reshape(t, [batch, -1, c, w, h])
    out = rearrange(out, "b t c h w -> b c t h w")
    return out


def pack_time(t):
    out = rearrange(t, "b c t h w -> b t c h w")
    _, _, c, w, h = out.size()
    return torch.reshape(out, [-1, c, w, h])


class TimeDownsample2x(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        kernel_size=3,
    ):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(dim, dim_out, kernel_size, stride=2)

    def forward(self, x):
        x = rearrange(x, "b c t h w -> b h w c t")
        b, h, w, c, t = x.size()
        x = torch.reshape(x, [-1, c, t])

        x = F.pad(x, self.time_causal_padding)
        out = self.conv(x)

        out = torch.reshape(out, [b, h, w, c, t])
        out = rearrange(out, "b h w c t -> b c t h w")
        out = rearrange(out, "b h w c t -> b c t h w")
        return out


class TimeUpsample2x(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        conv = nn.Conv1d(dim, dim_out * 2, 1)

        self.net = nn.Sequential(
            nn.SiLU(), conv, Rearrange("b (c p) t -> b c (t p)", p=2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, t = conv.weight.shape
        conv_weight = torch.empty(o // 2, i, t)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 2) ...")

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        x = rearrange(x, "b c t h w -> b h w c t")
        b, h, w, c, t = x.size()
        x = torch.reshape(x, [-1, c, t])

        out = self.net(x)
        out = out[:, :, 1:].contiguous()

        out = torch.reshape(out, [b, h, w, c, t])
        out = rearrange(out, "b h w c t -> b c t h w")
        return out


class AttnBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 max_seq_len=100,
                 mode='spatial',
                 fan_mode=False):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

        if fan_mode:
            from .multi_head_attention import FANLayer
            self.q_fan = FANLayer(input_dim=in_channels, output_dim=in_channels)
            self.k_fan = FANLayer(input_dim=in_channels, output_dim=in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        for layer in [self.q, self.k, self.v, self.proj_out]:
            nn.init.xavier_uniform_(layer.weight)  #
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)  #

    def forward(self, x, freqs_cis=None):
        h_ = x # self attn:(B*dyna_L, C,H,W); cross_attn:(B*)
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw

        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

class CxAttnBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 resolution=[32, 32],  # (32, 32),
                 context_resolutions=[32, 32],
                 kv_frames=1
                 ):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

        self.kv_frames = kv_frames
        self.kv_pos_emb = nn.parameter.Parameter(torch.zeros((kv_frames * context_resolutions[0] * context_resolutions[1], in_channels)),
                                                 requires_grad=True)
        #
        self.q_pos_emb = nn.parameter.Parameter(torch.zeros((kv_frames * resolution[0] * resolution[1], in_channels)),
                                                requires_grad=True)
        self.resolution = resolution
        self.context_resolutions = context_resolutions
    def set_kv_frames(self, kv_frames):
        self.kv_pos_emb.data = nn.parameter.Parameter(torch.zeros((kv_frames * self.context_resolutions[0] * self.context_resolutions[1], self.in_channels)),
                                                 requires_grad=True)
        self.kv_frames = kv_frames
    def forward(self, x, y):#x,y same shape
        # Normalize inputs
        # h_ = self.norm(x)  # Query input
        # y = self.norm(y)  # Key/Value input
        #
        # # Compute Q, K, V
        # q = self.q(h_)
        # k = self.k(y)
        # v = self.v(y)
        #
        # # compute attention
        # b, c, h, w = q.shape
        # q = q.reshape(b, c, h * w)
        # q = q.permute(0, 2, 1)  # b,hw,c
        # k = k.reshape(b, c, h * w)  # b,c,hw
        # w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # w_ = w_ * (int(c) ** (-0.5))
        # w_ = torch.nn.functional.softmax(w_, dim=2)
        #
        # # attend to values
        # v = v.reshape(b, c, h * w)
        # w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        # h_ = h_.reshape(b, c, h, w)
        #
        # h_ = self.proj_out(h_)
        """
                Args:
                    x: Input tensor [B, C, H, W] (used for Query)
                    y: Input tensor [B, C, H2, W2] (used for Key and Value)
                """
        B, C, H, W = x.shape
        H2, W2 = y.shape[-2:]
        assert H == self.resolution[0] and W == self.resolution[1], "Input resolution must match positional embedding size."

        T = y.shape[1]
        # Normalize inputs
        h_ = self.norm(x)  # Query input
        y = self.norm(y.flatten(0, 1)).unflatten(0, (B, T))  # Key/Value input
        # if self.kv_frames > 1:
            # B, t, C, H, W -> B, C, tH, W
        y = y.permute(0, 2, 1, 3, 4).reshape(B, C, -1, W2)
        # Compute Q, K, V
        q = self.q(h_)
        k = self.k(y)
        v = self.v(y)

        # Add positional embeddings
        q = q.view(B, C, -1).permute(0, 2, 1).contiguous()  # [B, HW, C]
        k = k.view(B, C, -1).permute(0, 2, 1).contiguous()  # [B, tHW, C]
        v = v.view(B, C, -1).permute(0, 2, 1).contiguous()  # [B, tHW, C]

        q = q + self.q_pos_emb.unsqueeze(0)  # Add Q position embedding
        k = k + self.kv_pos_emb.unsqueeze(0)  # Add KV position embedding

        # Compute attention
        attn_weights = torch.bmm(q, k.transpose(1, 2))  # [B, HW, tHW]
        attn_weights = attn_weights * (C ** -0.5)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # Attend to values
        attn_output = torch.bmm(attn_weights, v)  # [B, HW, C]
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]

        # Final projection and residual connection
        h_ = self.proj_out(attn_output)
        return x + h_
class TimeAttention(AttnBlock):
    def __init__(self, in_channels):
        super(TimeAttention, self).__init__(in_channels=in_channels, mode='temporal')
    def forward(self, x, time_ids=None, *args, **kwargs):
        x = rearrange(x, "b t c h w -> b h w t c")
        b, h, w, t, c = x.size()
        x = torch.reshape(x, (-1, t, c))

        x = super().forward(x, freqs_cis=time_ids, *args, **kwargs)

        x = torch.reshape(x, [b, h, w, t, c])
        return rearrange(x, "b h w t c -> b t c h w")


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


class Residual3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_t=16, stride_t=16, padding=0, pool_kernel_size=(16, 1, 1)):
        super(Residual3DConv, self).__init__()
        stride = (stride_t, 1, 1)
        kernel_size = (kernel_size_t, 1, 1)
        #
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        # #
        # if in_channels != out_channels:
        #     self.match_channels = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        # else:
        #     self.match_channels = None
        #
        # #
        # self.max_pool = nn.MaxPool3d(pool_kernel_size, stride=pool_kernel_size,  return_indices=True)

    def forward(self, x, p):
        #
        # residual = x #(B,C,T,H,W)

        d = x.permute(0, 2, 3, 1).unfold(1, p, p).unfold(2, p, p).permute(0, 1, 2, 4, 5, 3).contiguous()
        d = d.reshape(d.shape[0], d.shape[1], d.shape[2], -1, d.shape[-1]).permute(0, 4, 3, 1,2).contiguous()  # [B, H/P, W/P, P*P*C] -> [B,C , P*P, H/P, W/P]
        p_h, p_w = d.shape[-2:]




        #
        d = self.conv(d)#(B,C,T,H,W)->(B,C,1,H,W)

        #
        # if self.match_channels is not None:
        #     residual = self.match_channels(residual)

        #
        # x = x + self.max_pool(residual)

        return x

class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        pad_mode="constant",
        **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = (
            width_pad,
            width_pad,
            height_pad,
            height_pad,
            time_pad,
            0,
        )

        stride = (stride, 1, 1)#stride T_stride
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(
            chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs
        )

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"

        x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        return self.conv(x)


def ResnetBlockCausal3D(
    dim, kernel_size: Union[int, Tuple[int, int, int]], pad_mode: str = "constant"
):
    net = nn.Sequential(
        Normalize(dim),
        nn.SiLU(),
        CausalConv3d(dim, dim, kernel_size, pad_mode),
        Normalize(dim),
        nn.SiLU(),
        CausalConv3d(dim, dim, kernel_size, pad_mode),
    )
    return Residual(net)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        else:
            self.temb_proj = None
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
