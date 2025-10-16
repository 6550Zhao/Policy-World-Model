from typing import *

import torch
import torch.nn as nn
from .common_modules import nonlinearity
from diffusers.models.activations import get_activation
from .vae import Encoder, Decoder
from .multi_head_attention import MultiHeadAttention
class AttentionBlock(nn.Module):

    def __init__(
        self,
        channels,
        resolution=[32, 32],#(32, 32),
        norm_group=32,
        act_fn: str = "silu",
        num_head=8,
        dropout=0.05,
        kv_frames=1
    ) -> None:
        super().__init__()

        self.att = nn.MultiheadAttention(channels, num_head, dropout=dropout, batch_first=True)
        # self.att = MultiHeadAttention(channels, num_head, bias=True)
        self.resid_dropout = nn.Dropout(dropout)
        self.kv_norm = nn.GroupNorm(norm_group, channels, eps=1e-6)
        self.q_norm = nn.GroupNorm(norm_group, channels, eps=1e-6)
        self.kv_frames = kv_frames
        self.kv_pos_emb = nn.parameter.Parameter(torch.zeros((kv_frames * resolution[0] * resolution[1], channels)),
                                                 requires_grad=True)
        #
        self.q_pos_emb = nn.parameter.Parameter(torch.zeros((kv_frames * resolution[0] * resolution[1], channels)), requires_grad=True)
        self.resolution = resolution
        self.channels = channels
        self.num_head = num_head
        self.act = get_activation(act_fn)
    def set_kv_frames(self, kv_frames):
        self.kv_pos_emb.data = nn.parameter.Parameter(torch.zeros((kv_frames * self.resolution[0] * self.resolution[1], self.channels)),
                                                 requires_grad=True)
        self.kv_frames = kv_frames

    def forward(self, z, addin=None, attn_mask=None):
        # addin: [B, C, H, W] or [B, t, C, H, W]
        if self.kv_frames > 1:
            # B, t, C, H, W -> B, C, tH, W
            addin = addin.permute(0, 2, 1, 3, 4).reshape(addin.shape[0], addin.shape[2], -1, addin.shape[3])
        if addin is None:
            addin = z
        kv = self.kv_norm(addin).permute(0, 2, 3, 1).reshape(addin.shape[0], -1, addin.shape[1])  # [B, tHW, C]
        kv = kv + self.kv_pos_emb#(B, tHW, C)
        q = self.q_norm(z).permute(0, 2, 3, 1).reshape(z.shape[0], -1, z.shape[1])  # [B, HW, C]
        q = q + self.q_pos_emb
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(1, self.num_head, q.shape[1], kv.shape[1]).reshape(-1, q.shape[1], kv.shape[1])  # ->[B,n_head=8,HW=1024,tHW]
        attn_output, _ = self.att(q, kv, kv, attn_mask=attn_mask)
        attn_output = self.resid_dropout(attn_output)

        attn_output = attn_output.permute(0, 2, 1).reshape(z.shape)  # [B, C, H, W]
        # z = z + self.act(attn_output)
        self.act(z + attn_output)
        return z + attn_output


class ConditionalEncoder(Encoder):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        max_att_resolution=32,
        init_resolution=256,
        context_length=1,
    ):
        super().__init__(
            in_channels,
            out_channels,
            down_block_types,
            block_out_channels,
            layers_per_block,
            norm_num_groups,
            act_fn,
            double_z,
            mid_block_add_attention,
        )
        resolution = init_resolution
        self.max_att_resolution = max_att_resolution

        self.cross_att_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if not is_final_block:
                resolution //= 2

            if resolution <= max_att_resolution:
                self.cross_att_blocks.append(
                    AttentionBlock(output_channel, resolution, kv_frames=context_length))

    def set_context_length(self, context_length):
        for cross_att_block in self.cross_att_blocks:
            cross_att_block.set_kv_frames(context_length)

    def forward(
        self,
        sample: torch.FloatTensor,
        cond_features: List[torch.FloatTensor]
    ) -> torch.FloatTensor:

        sample = self.conv_in(sample)

        # down
        att_idx = 0
        for i, down_block in enumerate(self.down_blocks):
            sample = down_block(sample)
            if sample.shape[-1] <= self.max_att_resolution:
                sample = self.cross_att_blocks[att_idx](sample, cond_features[i + 1])
                att_idx += 1

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class ConditionalDecoder(Decoder):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
        max_att_resolution=32,
        init_resolution=16,
        context_length=1,
    ):
        super().__init__(
            in_channels,
            out_channels,
            up_block_types,
            block_out_channels,
            layers_per_block,
            norm_num_groups,
            act_fn,
            norm_type,
            mid_block_add_attention,
        )
        resolution = init_resolution
        self.max_att_resolution = max_att_resolution

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        self.cross_att_blocks = nn.ModuleList([
            AttentionBlock(output_channel, resolution, kv_frames=context_length)
        ])

        for i, up_block_type in enumerate(up_block_types):
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            if not is_final_block:
                resolution *= 2

            if resolution <= max_att_resolution:
                self.cross_att_blocks.append(
                    AttentionBlock(output_channel, resolution, kv_frames=context_length))

    def set_context_length(self, context_length):
        for cross_att_block in self.cross_att_blocks:
            cross_att_block.set_kv_frames(context_length)

    def forward(
        self,
        sample: torch.FloatTensor,
        cond_features: List[torch.FloatTensor],
    ) -> torch.FloatTensor:

        sample = self.conv_in(sample)
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        # middle
        sample = self.mid_block(sample, None)
        sample = sample.to(upscale_dtype)

        sample = self.cross_att_blocks[0](sample, cond_features[1])

        # up
        for i, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, None)
            if sample.shape[-1] <= self.max_att_resolution:
                sample = self.cross_att_blocks[i + 1](sample, cond_features[i + 2])

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
