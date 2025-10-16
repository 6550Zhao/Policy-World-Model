from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from .common_modules import *
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .misc import *
import math
from .conditional_vae import AttentionBlock
from .common_modules import CxAttnBlock, AttnBlock
class Updateable:
    def do_update_step(
            self, epoch: int, global_step: int, on_load_weights: bool = False
    ):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except:
                continue  # ignore attributes like property, which can't be retrived using getattr?
            if isinstance(module, Updateable):
                module.do_update_step(
                    epoch, global_step, on_load_weights=on_load_weights
                )
        self.update_step(epoch, global_step, on_load_weights=on_load_weights)

    def do_update_step_end(self, epoch: int, global_step: int):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except:
                continue  # ignore attributes like property, which can't be retrived using getattr?
            if isinstance(module, Updateable):
                module.do_update_step_end(epoch, global_step)
        self.update_step_end(epoch, global_step)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # override this method to implement custom update logic
        # if on_load_weights is True, you should be careful doing things related to model evaluations,
        # as the models and tensors are not guarenteed to be on the same device
        pass

    def update_step_end(self, epoch: int, global_step: int):
        pass

class VQGANEncoder(ModelMixin, ConfigMixin):
    @dataclass
    class Config:
        ch: int = 128
        ch_mult: List[int] = field(default_factory=lambda: [1, 2, 2, 4, 4])
        num_res_blocks: List[int] = field(default_factory=lambda: [4, 3, 4, 3, 4])
        attn_resolutions: List[int] = field(default_factory=lambda: [5])
        dropout: float = 0.0
        in_ch: int = 3
        out_ch: int = 3
        resolution: int = 256
        z_channels: int = 13
        double_z: bool = False

    def __init__(self,
                 ch: int = 128,
                 ch_mult: List[int] = [1, 2, 2, 4, 4],
                 num_res_blocks: List[int] = [4, 3, 4, 3, 4],
                 attn_resolutions: List[int] = [5],
                 dropout: float = 0.0,
                 in_ch: int = 3,
                 out_ch: int = 3,
                 resolution: int = 256,
                 z_channels: int = 13,
                 max_att_resolution: int = 32,
                 double_z: bool = False):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        self.max_att_resolution = max_att_resolution
        # downsampling
        self.conv_in = torch.nn.Conv2d(
            self.in_ch, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, True)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle,可以用来condition?
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )


        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        # for param in self.parameters():
        #     broadcast(param, src=0)

    def forward(self, x, return_features=False):
        # timestep embedding
        temb = None
        cond_features = []

        # downsampling
        hs = [self.conv_in(x)]#B,128,H,W
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
                if hs[-1].shape[-2] <= self.max_att_resolution:
                    cond_features.append(hs[-1])
        # middle
        h = hs[-1]#b,512,H/16,W/16
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        cond_features.append(h)
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)#(B,C1,H,W)-(B,C2,H,W)
        if return_features:
            return h, cond_features
        else:
            return h

class VQGANEncoder3D(ModelMixin, ConfigMixin):
    @dataclass
    class Config:
        ch: int = 128
        ch_mult: List[int] = field(default_factory=lambda: [1, 2, 2, 4, 4])
        num_res_blocks: List[int] = field(default_factory=lambda: [4, 3, 4, 3, 4])
        attn_resolutions: List[int] = field(default_factory=lambda: [5])
        dropout: float = 0.0
        in_ch: int = 3
        out_ch: int = 3
        resolution: int = 256
        z_channels: int = 13
        double_z: bool = False

    def __init__(self,
                 ch: int = 128,
                 ch_mult: List[int] = [1, 2, 2, 4, 4],
                 num_res_blocks: List[int] = [4, 3, 4, 3, 4],
                 attn_resolutions: List[int] = [5],
                 dropout: float = 0.0,
                 in_ch: int = 3,
                 out_ch: int = 3,
                 resolution: int = 256,
                 z_channels: int = 13,
                 max_att_resolution: int = 32,
                 double_z: bool = False):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        self.max_att_resolution = max_att_resolution
        # downsampling
        # self.conv_in = torch.nn.Conv2d(
        #     self.in_ch, self.ch, kernel_size=3, stride=1, padding=1)
        self.conv_first = CausalConv3d(self.in_ch, self.ch, kernel_size=(3, 3, 3), use_bias=True,
                                       pad_mode="constant")
        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlockCausal3D(
                        dim=block_in,
                        kernel_size=(3, 3, 3),
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, True)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle,可以用来condition?
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )


        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        # for param in self.parameters():
        #     broadcast(param, src=0)

    def forward(self, x, return_features=False):
        # timestep embedding
        temb = None
        cond_features = []

        # downsampling
        hs = [self.conv_in(x)]#B,128,H,W
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
                if hs[-1].shape[-2] <= self.max_att_resolution:
                    cond_features.append(hs[-1])
        # middle
        h = hs[-1]#b,512,H/16,W/16
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        cond_features.append(h)
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)#(B,C1,H,W)-(B,C2,H,W)
        if return_features:
            return h, cond_features
        else:
            return h
class ConditionalVQGANEncoder(VQGANEncoder):
    def __init__(
        self,
        f_resolutions: List[int] = [[16,28], [16,28]], #[[32,32], [16,16], [16,16]],#[32, 16, 16],
        context_resolutions: List[int] = [[16, 28], [16,28]],
        block_out_channels: List[int] = [256, 512, 512],
        max_att_resolution=16, #32,
        init_resolution=256,
        context_length=1,
    ):
        super().__init__(max_att_resolution=max_att_resolution)
        resolution = init_resolution
        self.max_att_resolution = max_att_resolution

        self.cross_att_blocks = nn.ModuleList([])
        self.sf_att_blocks = nn.ModuleList([])
        self.post_sf_att_blocks = nn.ModuleList([])
        # down
        j=0
        for i, output_channel in enumerate(block_out_channels):
            output_channel = block_out_channels[i]
            if i >= len(block_out_channels)-len(f_resolutions):
                resolution = f_resolutions[j]
                c_resolutions = context_resolutions[j]
                if resolution[0] <= max_att_resolution:
                    # self.cross_att_blocks.append(AttentionBlock(output_channel, resolution, kv_frames=context_length))
                    # self.sf_att_blocks.append(AttentionBlock(output_channel, resolution))
                    self.cross_att_blocks.append(CxAttnBlock(output_channel, resolution, c_resolutions, kv_frames=context_length))
                    self.sf_att_blocks.append(AttnBlock(output_channel, fan_mode=False))
                    self.post_sf_att_blocks.append(AttnBlock(output_channel, fan_mode=False))
                j += 1


    def set_context_length(self, context_length):
        for cross_att_block in self.cross_att_blocks:
            cross_att_block.set_kv_frames(context_length)

    def forward(
        self,
        sample: torch.FloatTensor,
        cond_features: List[torch.FloatTensor],
        cond=False,
        attn_mask=None,
        time_ids=None,
    ) -> torch.FloatTensor:
        # timestep embedding
        temb = None
        att_idx = 0
        # downsampling
        hs = [self.conv_in(sample)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
                #CXT
                if hs[-1].shape[-2] <= self.max_att_resolution and cond:
                    h = self.sf_att_blocks[att_idx](hs[-1])
                    # h = self.sf_att_blocks[att_idx](hs[-1], hs[-1])
                    h = self.cross_att_blocks[att_idx](h, cond_features[att_idx-len(self.cross_att_blocks)])
                    h = self.post_sf_att_blocks[att_idx](h)
                    hs.append(h)
                    att_idx += 1
        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        #----------------------------------------------------------CXT
        if cond:
            h = self.sf_att_blocks[att_idx](h, h) # h = self.sf_att_blocks[att_idx](h)
            h = self.cross_att_blocks[att_idx](h, cond_features[att_idx-len(self.cross_att_blocks)])
            h = self.post_sf_att_blocks[att_idx](h, h)
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # h = self.quant_conv(h)
        return h

class LFQuantizer(nn.Module):
    def __init__(self, num_codebook_entry: int = -1,
                 codebook_dim: int = 13,
                 beta: float = 0.25,
                 entropy_multiplier: float = 0.1,
                 commit_loss_multiplier: float = 0.1, ):
        super().__init__()
        self.codebook_size = 2 ** codebook_dim
        print(
            f"Look-up free quantizer with codebook size: {self.codebook_size}"
        )
        self.e_dim = codebook_dim
        self.beta = beta

        indices = torch.arange(self.codebook_size)

        binary = (
                         indices.unsqueeze(1)
                         >> torch.arange(codebook_dim - 1, -1, -1, dtype=torch.long)
                 ) & 1

        embedding = binary.float() * 2 - 1
        self.register_buffer("embedding", embedding)
        self.register_buffer(
            "power_vals", 2 ** torch.arange(codebook_dim - 1, -1, -1)
        )
        self.commit_loss_multiplier = commit_loss_multiplier
        self.entropy_multiplier = entropy_multiplier

    def get_indices(self, z_q):
        return (
            (self.power_vals.reshape(1, -1, 1, 1) * (z_q > 0).float())
            .sum(1, keepdim=True)
            .long()
        )

    def get_codebook_entry(self, indices, shape=None):
        if shape is None:
            h, w = int(math.sqrt(indices.shape[-1])), int(math.sqrt(indices.shape[-1]))
        else:
            h, w = shape
        b, _ = indices.shape
        indices = indices.reshape(-1)
        z_q = self.embedding[indices]
        z_q = z_q.view(b, h, w, -1)

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

    def get_latent_entry(self, latent, shape=None):
        if shape is None:
            h, w = int(math.sqrt(latent.shape[-2])), int(math.sqrt(latent.shape[-2]))
        else:
            h, w = shape
        b, _, c = latent.shape
        # indices = latent.reshape(-1)
        z_latent = torch.tanh((latent @ self.embedding[None, ...]))
        z_latent = z_latent.view(b, h, w, -1)

        # reshape back to match original input shape
        z_latent = z_latent.permute(0, 3, 1, 2).contiguous()

        return z_latent
    def forward(self, z, get_code=False):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        if get_code:
            return self.get_codebook_entry(z)

        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        ge_zero = (z_flattened > 0).float()
        ones = torch.ones_like(z_flattened)
        z_q = ones * ge_zero + -ones * (1 - ge_zero)

        # preserve gradients
        z_q = z_flattened + (z_q - z_flattened).detach()

        # compute entropy loss
        CatDist = torch.distributions.categorical.Categorical
        logit = torch.stack(
            [
                -(z_flattened - torch.ones_like(z_q)).pow(2),
                -(z_flattened - torch.ones_like(z_q) * -1).pow(2),
            ],
            dim=-1,
        )
        cat_dist = CatDist(logits=logit)
        entropy = cat_dist.entropy().mean()
        mean_prob = cat_dist.probs.mean(0)
        mean_entropy = CatDist(probs=mean_prob).entropy().mean()

        # compute loss for embedding
        commit_loss = torch.mean(
            (z_q.detach() - z_flattened) ** 2
        ) + self.beta * torch.mean((z_q - z_flattened.detach()) ** 2)
        # logi_loss = -torch.mean(z_q.detach()*z_flattened)
        # reshape back to match original input shape
        z_q = z_q.view(z.shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return {
            "z": z_q,
            "quantizer_loss": commit_loss * self.commit_loss_multiplier,#commitment losses
            "entropy_loss": (entropy - mean_entropy) * self.entropy_multiplier,#entropy penalty
            "indices": self.get_indices(z_q),
        }


class VQGANDecoder(ModelMixin, ConfigMixin):
    def __init__(self, ch: int = 128,
                 ch_mult: List[int] = [1, 1, 2, 2, 4],
                 num_res_blocks: List[int] = [4, 4, 3, 4, 3],
                 attn_resolutions: List[int] = [5],
                 dropout: float = 0.0,
                 in_ch: int = 3,
                 out_ch: int = 3,
                 resolution: int = 256,
                 z_channels: int = 13,
                 double_z: bool = False,
                 max_att_resolution: int = 32):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        self.give_pre_end = False
        self.max_att_resolution = max_att_resolution
        self.z_channels = z_channels
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = self.resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # print(
        #     "Working with z of shape {} = {} dimensions.".format(
        #         self.z_shape, np.prod(self.z_shape)
        #     )
        # )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, True)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )
        self.post_quant_conv = torch.nn.Conv2d(
            z_channels, z_channels, 1
        )


    def forward(self, z, return_features=False):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape
        # timestep embedding
        temb = None
        output = dict()
        cond_features = []
        z = self.post_quant_conv(z)

        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        cond_features.append(h)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                if h.shape[-2] <=self.max_att_resolution:
                    cond_features.append(h)
        # end
        output["output"] = h
        if self.give_pre_end:
            if return_features:
                return output, cond_features
            return output

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        output["output"] = h
        if return_features:
            return output, cond_features
        return output

class VQGANDecoder3D(ModelMixin, ConfigMixin):
    def __init__(self, ch: int = 128,
                 ch_mult: List[int] = [1, 1, 2, 2, 4],
                 num_res_blocks: List[int] = [4, 4, 3, 4, 3],
                 attn_resolutions: List[int] = [5],
                 dropout: float = 0.0,
                 in_ch: int = 3,
                 out_ch: int = 3,
                 resolution: int = 256,
                 z_channels: int = 13,
                 double_z: bool = False,
                 max_att_resolution: int = 32):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        self.give_pre_end = False
        self.max_att_resolution = max_att_resolution
        self.z_channels = z_channels
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = self.resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, True)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )
        self.post_quant_conv = torch.nn.Conv2d(
            z_channels, z_channels, 1
        )


    def forward(self, z, return_features=False):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape
        # timestep embedding
        temb = None
        output = dict()
        cond_features = []
        z = self.post_quant_conv(z)

        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        cond_features.append(h)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                if h.shape[-2] <=self.max_att_resolution:
                    cond_features.append(h)
        # end
        output["output"] = h
        if self.give_pre_end:
            if return_features:
                return output, cond_features
            return output

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        output["output"] = h
        if return_features:
            return output, cond_features
        return output
class ConditionalVQGANDecoder(VQGANDecoder):

    def __init__(
        self,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: List[int] = [512, 512],
        max_att_resolution=32,
        init_resolution: List[int] = [[16,28], [32,56]], #[[16,16], [32,32]],#[16, 32],
        context_resolutions: List[int] = [[16,28], [32,56]],
        context_length=1,
    ):
        super().__init__(max_att_resolution=max_att_resolution)
        resolution = init_resolution
        self.max_att_resolution = max_att_resolution
        self.cross_att_blocks = nn.ModuleList()
        self.sf_att_blocks = nn.ModuleList([])
        self.post_sf_att_blocks = nn.ModuleList([])
        for i, output_channel in enumerate(block_out_channels):
            if resolution[i][0] <= max_att_resolution:
                # self.cross_att_blocks.append(AttentionBlock(output_channel, resolution[i], kv_frames=context_length))
                # self.sf_att_blocks.append(AttentionBlock(output_channel, resolution[i]))
                self.cross_att_blocks.append(CxAttnBlock(output_channel, resolution[i], context_resolutions[i], kv_frames=context_length))
                self.sf_att_blocks.append(AttnBlock(output_channel, fan_mode=False))
                self.post_sf_att_blocks.append(AttnBlock(output_channel, fan_mode=False))
    def set_context_length(self, context_length):
        for cross_att_block in self.cross_att_blocks:
            cross_att_block.set_kv_frames(context_length)
    def forward(
        self,
        z: torch.FloatTensor,
        cond_features: List[torch.FloatTensor],
        cond=False,
        attn_mask=None,
    ) -> torch.FloatTensor:
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape
        # timestep embedding
        temb = None
        output = dict()

        h = self.conv_in(z)#2D
        i = 0
        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        #attention 1
        if cond:
            h = self.sf_att_blocks[i](h, h)
            h = self.cross_att_blocks[i](h, cond_features[i])#Q:(B(T-t),C,H,W),k,v:(B(T-t),t,C,H,W)
            h = self.post_sf_att_blocks[i](h, h)
            i += 1
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                if h.shape[-2] <= self.max_att_resolution and cond and i < 2:
                    #attention_2,3,4,...
                    h = self.sf_att_blocks[i](h, h)
                    h = self.cross_att_blocks[i](h, cond_features[i])
                    h = self.post_sf_att_blocks[i](h, h)
                    #self attn?
                    i += 1

        # end
        output["output"] = h
        if self.give_pre_end:
            return output

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        output["output"] = h
        return output
class MAGVITv2(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            configs=None,
            conditional=False,

    ):
        super().__init__()
        if conditional:
            self.encoder = ConditionalVQGANEncoder(f_resolutions = [[int(configs.dataset.params.resolution_h//16),int(configs.dataset.params.resolution_w//16)], [int(configs.dataset.params.resolution_h//16),int(configs.dataset.params.resolution_w//16)]],
                                                   max_att_resolution = int(configs.dataset.params.resolution_h//16))
            self.decoder = ConditionalVQGANDecoder(init_resolution= [[int(configs.dataset.params.resolution_h//16),int(configs.dataset.params.resolution_w//16)], [int(configs.dataset.params.resolution_h//8),int(configs.dataset.params.resolution_w//8)]],
                                                   max_att_resolution = int(configs.dataset.params.resolution_h//8))
        else:
            self.encoder = VQGANEncoder()
            self.decoder = VQGANDecoder()
        self.quantize = LFQuantizer()

    def forward(self, return_loss=False):
        pass

    def encode(self, pixel_values, return_loss=False):
        hidden_states = self.encoder(pixel_values)
        quantized_states = self.quantize(hidden_states)['z']
        codebook_indices = self.quantize.get_indices(quantized_states).reshape(pixel_values.shape[0], -1)
        output = (quantized_states, codebook_indices)
        return output

    def get_code(self, pixel_values, return_feature = False):
        hidden_states = self.encoder(pixel_values)#(batch,13,H,W)
        codebook_indices = self.quantize.get_indices(self.quantize(hidden_states)['z']).reshape(pixel_values.shape[0], -1)
        if return_feature:
            return codebook_indices, hidden_states
        else:
            return codebook_indices

    def decode_code(self, codebook_indices, shape=None):
        z_q = self.quantize.get_codebook_entry(codebook_indices, shape=shape)

        reconstructed_pixel_values = self.decoder(z_q)["output"]
        return reconstructed_pixel_values

if __name__ == '__main__':
    encoder = VQGANEncoder()
    import ipdb
    ipdb.set_trace()
    print()