from typing import *

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import os
import time
from safetensors.torch import load_file
from PIL import Image
import torchvision.transforms as transforms
from dataclasses import dataclass
from diffusers.models.autoencoders.vae import VectorQuantizer
from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from .multi_head_attention import FANLayer
from .vae import Encoder, Decoder, MLP
from .conditional_vae import ConditionalEncoder, ConditionalDecoder
from .modeling_magvitv2 import MAGVITv2
from .conditional_vae import AttentionBlock
import torch.nn.functional as F
SPECIAL_CONTEXT_TOKEN = -10
SPECIAL_DYNAMIC_TOKEN = -20
@dataclass
class CompressiveVQEncoderOutput(BaseOutput):

    latents: torch.FloatTensor
    dynamics_latents: torch.FloatTensor

def find_token_positions(tensor, token):
    return (tensor == token).nonzero(as_tuple=True)[-1]
@dataclass
class CompressiveVQDecoderOutput(BaseOutput):

    sample: torch.FloatTensor
    ref_sample: Optional[torch.FloatTensor] = None
    commit_loss: Optional[torch.FloatTensor] = None
    dyn_commit_loss: Optional[torch.FloatTensor] = None

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")
class Compressive_magvit_v2(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            latent_channels: int = 13,
            num_vq_embeddings: int = 256,
            num_dyn_embeddings: int = 256,
            config_exps: dict = None,
            max_att_resolution=32,
            patch_size=4,
            enable_mask_token: bool = False,
            conv_enable: bool = False
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.dyna_latent_channels = latent_channels
        self.context_length = config_exps.dataset.params.context_length
        self.num_vq_embeddings = num_vq_embeddings
        self.num_dyn_embeddings = num_dyn_embeddings
        self.patch_size = config_exps.model.vq_model.patch_size #h , w
        self.max_att_resolution = max_att_resolution
        self.latent_patch = config_exps.model.vq_model.latent_size
        self.context_vq_model = MAGVITv2(conditional=False)
        self.dynamic_vq_model = MAGVITv2(configs=config_exps, conditional=True)
        self.conv_enable = conv_enable
        # self.dy_quant_l_in = MLP(c_in, 4*c_in, c_out)
        self.enable_mask_token = enable_mask_token
        self.ctx_res = [16, 28]  # TODO: magic number
        self.dyn_res = [4, 7] # [8,14]
        if self.patch_size==4:
            self.dy_quant_l_in = nn.Linear(self.dyna_latent_channels * self.patch_size * self.patch_size, self.dyna_latent_channels)
            self.dy_quant_l_out = nn.Linear(self.dyna_latent_channels, self.dyna_latent_channels * self.patch_size * self.patch_size)
            nn.init.xavier_uniform_(self.dy_quant_l_in.weight)
            nn.init.zeros_(self.dy_quant_l_in.bias)
            nn.init.xavier_uniform_(self.dy_quant_l_out.weight)
            nn.init.zeros_(self.dy_quant_l_out.bias)
        elif self.patch_size==2:#28 tokens
            self.dy_quant_l_in = MLP(self.dyna_latent_channels * self.patch_size * self.patch_size, 512, self.dyna_latent_channels)
            self.dy_quant_l_out = MLP(self.dyna_latent_channels, 512, self.dyna_latent_channels * self.patch_size * self.patch_size)

        self.mask_prob = 0.25
        self.cond_enable = config_exps.model.cond_enable
        self.set_context_length()

    def forward_dy_quant_l_in(self, x):
        for layer in self.dy_quant_l_in:
            x = layer(x)
        return x

    def forward_dy_post_quant_l_out(self, x):
        for layer in self.dy_post_quant_l_out:
            x = layer(x)
        return x
    def init_modules(self,model=None, model_path=None, noise_std=0.01):
        if model == 'context_vq_model' and model_path is not None:
            from safetensors.torch import load_file
            weights = load_file(model_path)
            self.context_vq_model.load_state_dict(weights, strict=True)
        elif model == 'dynamic_vq_model' and model_path is not None:
            print(self.dynamic_vq_model.load_state_dict(self.context_vq_model.state_dict(), strict=False))
        else:
            pass

    def zero_init(self, parameters):
        pass

    def frozen_module(self, module=None):
        if module is not None:
            target_module = getattr(self, module, None)
            if target_module is not None:
                for _, p in target_module.named_parameters():
                    p.requires_grad = False
            else:
                raise ValueError(f"Module {module} not found in model.")

    def set_context_length(self):
        self.config_exps['context_length'] = self.context_length
        self.dynamic_vq_model.encoder.set_context_length(self.context_length)
        self.dynamic_vq_model.decoder.set_context_length(self.context_length)
    def context_process(self,F):
        B,fur_len,con_len,_,_,_ = F.shape
        f_0 = F[:, :fur_len//2, :self.context_length,:,:,:]
        f_1 = F[:, fur_len//2:, -self.context_length:,:,:,:]
        return torch.concat((f_0,f_1),1)
    @apply_forward_hook
    def tokenize(self,
                 pixel_values: torch.FloatTensor,
                 context_pixel_values: torch.FloatTensor=None,
                 context_length: int = 0,
                 special_token: dict = None,
                 only_context_vq: bool = False,
                 return_label: bool = True,
                 save_last: bool = False,
                 ):
        if not only_context_vq:#dynamic
            assert context_length == self.context_length  # TODO: fix
            if special_token is None:
                raise NotImplementedError("Special tokens are not implemented.")
            if context_pixel_values is not None:
                B, Td, C, H, W = pixel_values.shape
                _, Tc, _, Hc, Wc = context_pixel_values.shape
                context_frames = context_pixel_values.reshape(-1, C, Hc, Wc)
                future_frames = pixel_values.reshape(-1, C, H, W)
                future_length = Td #+ 1
            else:
                B, T, C, H, W = pixel_values.shape
                context_frames = pixel_values[:, :context_length].reshape(-1, C, H, W)
                future_frames = pixel_values[:, context_length-1:].reshape(-1, C, H, W)
                future_length = T - context_length+1
        else:#mmu
            B, C, H, W = pixel_values.shape
            context_frames = pixel_values
            context_length = 1
        # encode context frames
        h, cond_features = self.context_vq_model.encoder(context_frames, return_features=True)  # h->(T_current,C,H,W) # Bt,3,H,W->Bt,13,H//16,W//16,h.max=16
        if not only_context_vq:
            if self.context_length > 1:
                B = future_frames.shape[0] // future_length
                cond_features = [
                    self.context_process(f.reshape(B, Tc, *f.shape[-3:]).unsqueeze(1)
                    .repeat(1, future_length, 1, 1, 1, 1)).reshape(-1, self.context_length, *f.shape[-3:])
                    for f in cond_features]  # B*(T-t), t, C, H, W。reshape(-1, 3, *f.shape[-3:])
            else:
                cond_features = [
                    f.unsqueeze(1).repeat(1, future_length, 1, 1, 1).reshape(-1, *f.shape[-3:])
                    for f in cond_features]
        # context vq
        quant, commit_loss, _, info = self.context_vq_model.quantize(h).values()
        # for context frames ,one is history key frame and another is current frame
        if only_context_vq:
            indices_c = info.flatten(-2).reshape(B, context_length, -1)  # (b*c,1,h,w)->(b,c,h*w)
            token_soi = special_token['<|soi|>'].to(pixel_values.device)  # special token for context frames
            token_eoi = special_token['<|eoi|>'].to(pixel_values.device)
            token_soi = torch.ones(B, context_length, 1).to(indices_c.device, indices_c.dtype) * token_soi
            token_eoi = torch.ones(B, context_length, 1).to(indices_c.device, indices_c.dtype) * token_eoi
            indices_c = torch.cat([token_soi, indices_c, token_eoi], dim=2).reshape(B, -1)  # [:, 1:]
            indices = {'context': indices_c}
            labels = None
            return indices, labels
        else:
            indices_c = info.flatten(-2).reshape(B, Tc, -1)  # (b*c,1,h,w)->(b,c,h*w)
            token_soi = special_token['<|soi|>'].to(pixel_values.device)  # special token for context frames
            token_eoi = special_token['<|eoi|>'].to(pixel_values.device)
            token_soi = torch.ones(B, Tc, 1).to(indices_c.device, indices_c.dtype) * token_soi
            token_eoi = torch.ones(B, Tc, 1).to(indices_c.device, indices_c.dtype) * token_eoi
            indices_c = torch.cat([token_soi, indices_c, token_eoi], dim=2).reshape(B, -1)  # [:, 1:]
        d = self.dynamic_vq_model.encoder(future_frames, cond_features, cond=self.cond_enable, attn_mask=None, time_ids=None)  # d.max=23
        p = self.patch_size
        d = d.permute(0, 2, 3, 1).unfold(1, p, p).unfold(2, p, p).permute(0, 1, 2, 4, 5, 3).contiguous()  # patchify: [B, H/P, W/P, P, P, C]
        d = d.reshape(d.shape[0], d.shape[1], d.shape[2], -1).permute(0, 3, 1, 2).contiguous()#[B, H/P, W/P, P*P*C] -> [B, P*P*C, H/P, W/P]
        p_h, p_w = d.shape[-2:]
        d = self.dy_quant_l_in(d.flatten(-2).permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        d = d.reshape(d.shape[0], d.shape[1], p_h, p_w).contiguous()
        d = self.dynamic_vq_model.encoder.quant_conv(d)
        d_latent_size = d.shape[-2]*d.shape[-1]
        quant_d, dyn_commit_loss, _, info_d = self.dynamic_vq_model.quantize(d).values()

        indices_d = info_d.flatten(-2).reshape(B, future_length, -1) + self.num_vq_embeddings
        token_sod = special_token['<|sod|>'].to(pixel_values.device)  # special token for context frames
        token_eod = special_token['<|eod|>'].to(pixel_values.device)
        token_sod = torch.ones(B, future_length, 1).to(indices_d.device, indices_d.dtype) * token_sod
        token_eod = torch.ones(B, future_length, 1).to(indices_d.device, indices_d.dtype) * token_eod
        indices_d = torch.cat([token_sod, indices_d, token_eod], dim=2).reshape(B, -1)
        if return_label:
            indices = {'context': indices_c, 'dynamic': indices_d[:, :-(2+d_latent_size)]} #
            labels = {'dynamic': indices_d[:, 2+d_latent_size:].clone()}
        else:
            indices = {'context': indices_c, 'dynamic': indices_d[:, :-(2+d_latent_size)]}#
            labels = {'dynamic': indices_d[:, 2+d_latent_size:].clone()}
        return indices, labels
    @apply_forward_hook
    def detokenize(self,
                   indices,
                   batch_ids=None,
                   offset_tokenzier=0,
                   sptids_dict=None,
                   c_start_sod=None,
                   c_end_sod=None,
                   d_start_sod=None,
                   d_end_sod=None,
                   cache=None,
                   return_cache=False,
                   only_context_vq: bool = False,):

        B = indices['context'].shape[0]
        sod_token = sptids_dict['<|sod|>']
        eod_token = sptids_dict['<|eod|>']
        soi_token = sptids_dict['<|soi|>']
        eoi_token = sptids_dict['<|eoi|>']
        context_q = []
        dynamic_q = []
        if only_context_vq:#Todo only reconstruct context
            ctx_h, ctx_w = self.ctx_res
            indices_c = indices['context'] #B,T,L
            if c_start_sod is None:
                c_start_sod = torch.where(indices_c == soi_token.to(indices_c.device))[-1]
            if c_end_sod is None:
                c_end_sod = torch.where(indices_c == eoi_token.to(indices_c.device))[-1]
            context_q.append(indices_c[:, c_start_sod[0] + 1: c_end_sod[0]])
            context_q = self.context_vq_model.quantize.get_codebook_entry(context_q[0], shape=(ctx_h, 36))  # ->(B*Tc,C,H,W)
            with torch.no_grad():
                ref_dec, _ = self.context_vq_model.decoder(context_q, return_features=True)  # (B*Tc,C,H,W)
            return ref_dec

        if batch_ids is not None:
            assert B > batch_ids, "batch size must be larger than batch ids"
            B = len([batch_ids])
            indices_c = indices['context'][batch_ids]# (B,L)
            indices_d = indices['dynamic'][batch_ids]
        else:
            indices_c = indices['context']#(B,L)
            indices_d = indices['dynamic']
        if c_start_sod is None:
            c_start_sod = torch.where(indices_c == soi_token.to(indices_c.device))[-1]
        if d_start_sod is None:
            d_start_sod = torch.where(indices_d == sod_token.to(indices_c.device))[-1]
        if c_end_sod is None:
            c_end_sod = torch.where(indices_c == eoi_token.to(indices_c.device))[-1]
        if d_end_sod is None:
            d_end_sod = torch.where(indices_d == eod_token.to(indices_c.device))[-1]
        indices_c = (indices_c - offset_tokenzier).clamp(min=0, max=self.num_dyn_embeddings - 1)  # (B,Lc)
        indices_d = (indices_d - offset_tokenzier - self.num_vq_embeddings).clamp(min=0, max=self.num_dyn_embeddings - 1) #(B,ld) # Todo,avoid value overflow
        ctx_h, ctx_w = self.ctx_res  # TODO: magic number
        dyn_h, dyn_w = self.dyn_res

        context_length = len(c_start_sod)//B
        future_length = len(d_start_sod)//B
        for i in range(len(d_start_sod)//B):
            dynamic_q.append(indices_d[:, d_start_sod[i]+1: d_end_sod[i]].unsqueeze(1))
        dynamic_q = torch.cat(dynamic_q, 1).flatten(0, 1)#list:11(B,1,28)->(B*11, 28)

        #DECODER PART
        # decode context frames
        if cache is not None:
            ref_dec, cond_features = cache["context_dec"], cache["cond_features"]
        else:
            for i in range(len(c_start_sod)//B):
                context_q.append(indices_c[:, c_start_sod[i] + 1: c_end_sod[i]].unsqueeze(1))
            context_q = torch.cat(context_q, 1).flatten(0, 1)  #(B*(T-t), 448)
            context_q = self.context_vq_model.quantize.get_codebook_entry(context_q, shape=(ctx_h, ctx_w))#->(B*Tc,C,H,W)
            with torch.no_grad():
                ref_dec, cond_features = self.context_vq_model.decoder(context_q, return_features=True)
        dynamic_q = self.dynamic_vq_model.quantize.get_codebook_entry(dynamic_q, shape=(dyn_h, dyn_w))#->(B*Td,C,h,w)

        if self.context_length > 1:
            cond_features = [self.context_process(f.reshape(B, context_length, *f.shape[-3:]).unsqueeze(1).repeat(1, future_length, 1, 1, 1, 1)).reshape(-1, self.context_length, *f.shape[-3:]) for f in cond_features]
        else:
            cond_features = [f.unsqueeze(1).repeat(1, self.segment_len, 1, 1, 1).reshape(-1, *f.shape[-3:]).unsqueeze(1) for f in cond_features]

        dynamic_q = self.dynamic_vq_model.decoder.post_quant_conv(dynamic_q)
        dynamic_q = self.dy_quant_l_out(dynamic_q.flatten(-2).permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous() #->(B,C,H*W)
        dynamic_q = dynamic_q.reshape(dynamic_q.shape[0], dynamic_q.shape[1], dyn_h, dyn_w).contiguous()

        C = self.dyna_latent_channels
        dynamic_q = dynamic_q.permute(0, 2, 3, 1).reshape(dynamic_q.shape[0], dynamic_q.shape[2], dynamic_q.shape[3], self.patch_size,
                                                  self.patch_size, C).contiguous()
        dynamic_q = torch.einsum("nhwpqc->nchpwq", dynamic_q)
        dynamic_q = torch.reshape(dynamic_q, [dynamic_q.shape[0], C, self.patch_size*dyn_h, self.patch_size*dyn_w])
        with torch.no_grad():
            dec = self.dynamic_vq_model.decoder(dynamic_q, cond_features, cond=self.cond_enable, attn_mask=None)
        _, c, hc, wc = ref_dec['output'].shape
        _, c, h, w = dec['output'].shape
        dec = dec['output'].reshape(B, future_length, c, h, w)
        if hc != h and wc != w:
            ref_dec = F.interpolate(ref_dec['output'], size=(h, w), mode='bilinear', align_corners=False)
            ref_dec = ref_dec.reshape(B, context_length, c, h, w)
        else:
            ref_dec = ref_dec['output'].reshape(B, context_length, c, hc, wc)

        if return_cache:
            return torch.cat([ref_dec, dec], dim=1), {"context_dec": ref_dec, "cond_features": cond_features}
        else:
            return torch.cat([ref_dec, dec], dim=1), {"context_len": self.context_length, "dynamic_len": future_length}


    def forward(
        self,
        sample: torch.FloatTensor,
        return_dict: bool = True,
        return_loss: bool = False,
        segment_len: int = None,
        time_step: List = None,
        dyn_sample: torch.FloatTensor = None,
        train: bool = True,
        mask_token = False,
    ) -> Union[CompressiveVQDecoderOutput, Tuple[torch.FloatTensor, ...]]:
        # condition encoder
        self.segment_len = segment_len

        # self.mask_prob

        if train:
            attn_mask = (torch.rand(dyn_sample.shape[0])[...,None, None, None] < -1).to(sample.device)  # [B, 1, 1, 1]
        else:
            attn_mask = (torch.rand(dyn_sample.shape[0])[...,None, None, None] < -1).to(sample.device)

        h, cond_features = self.context_vq_model.encoder(sample, return_features=True)#Bt,3,H,W->Bt,13,H//16,W//16,h.max=16
        if self.context_length > 1:
            B = dyn_sample.shape[0] // self.segment_len
            cond_features = [f.reshape(B, self.context_length, *f.shape[-3:]).unsqueeze(1).expand(-1, self.segment_len, -1, -1, -1, -1).reshape(-1, self.context_length, *f.shape[-3:])
                            for f in cond_features]  # B, t, C, H, W -> B*(T-t), t, C, H, W
            if time_step:
                time_step = [b_time_ids[:self.context_length]+[b_time_ids[self.context_length+i]] for b_time_ids in time_step for i in range(self.segment_len)]
        else:
            cond_features = [f.unsqueeze(1).repeat(1, self.segment_len, 1, 1, 1).reshape(-1,*f.shape[-3:]).unsqueeze(1) for f in cond_features]

        h, commit_loss, entropy_loss, context_codebook_indices = self.context_vq_model.quantize(h).values()#h,:(b,t, c,h,w),context

        d_out = self.dynamic_vq_model.encoder(dyn_sample, cond_features, cond=self.cond_enable, attn_mask=attn_mask, time_ids=torch.tensor(time_step).to(sample.device) if time_step else None) #d.max=23
        p = self.patch_size
        d = d_out.permute(0, 2, 3, 1).unfold(1, p, p).unfold(2, p, p).permute(0, 1, 2, 4, 5, 3).contiguous()  # [B, H, W, C]->[B, H/P, W/P, P, P, C]
        d = d.reshape(d.shape[0], d.shape[1], d.shape[2], -1).permute(0, 3, 1, 2).contiguous()#[B, H/P, W/P, P*P*C] -> [B, P*P*C, H/P, W/P]
        p_h, p_w = d.shape[-2:]
        d = self.dy_quant_l_in(d.flatten(-2).permute(0, 2, 1).contiguous()).permute(0,2,1).contiguous()

        d = d.reshape(d.shape[0], d.shape[1], p_h, p_w).contiguous()
        d_h = self.dynamic_vq_model.encoder.quant_conv(d)#.transpose(-1, -2).unsqueeze(-1)#[B, P*P*C, H/P, W/P]->[B, C, H/P, W/P],d.max=67
        dyn_h, dyn_commit_loss, dyn_entropy_loss, dynamic_codebook_indices = self.dynamic_vq_model.quantize(d_h).values()
        H, W, P, C = h.shape[-2], h.shape[-1], self.patch_size, self.dyna_latent_channels
        dyn_h = self.dynamic_vq_model.decoder.post_quant_conv(dyn_h)
        # dyn_h = self.post_quant_linear(dyn_h.squeeze(-1).transpose(-1, -2))
        dyn_h = self.dy_quant_l_out(dyn_h.flatten(-2).permute(0, 2, 1).contiguous()).permute(0,2,1).contiguous()
        dyn_h = dyn_h.reshape(dyn_h.shape[0], dyn_h.shape[1], p_h, p_w).contiguous()

        dyn_h = dyn_h.permute(0,2,3,1).reshape(dyn_h.shape[0], dyn_h.shape[2], dyn_h.shape[3], self.patch_size, self.patch_size, C).contiguous()
        dyn_h = torch.einsum("nhwpqc->nchpwq", dyn_h)
        dyn_h = torch.reshape(dyn_h, [dyn_h.shape[0], C, p_h*P, p_w*P]) #+ self.dynamic_vq_model.decoder.post_quant_conv(self.dynamic_vq_model.encoder.quant_conv(d_out))
        #---------------------
        #DECODER PART
        ref_dec, cond_features = self.context_vq_model.decoder(h, return_features=True)#(B,C,H,W)
        if self.context_length > 1:
            B = dyn_h.shape[0] // self.segment_len
            cond_features = [
                f.reshape(B, self.context_length, *f.shape[-3:]).unsqueeze(1).expand(-1, self.segment_len, -1, -1, -1, -1).reshape(-1, self.context_length, *f.shape[-3:]) for f in cond_features
            ]  # B*(T-t), t, C, H, W
        else:
            cond_features = [f.unsqueeze(1).repeat(1, self.segment_len, 1, 1, 1).reshape(-1,*f.shape[-3:]).unsqueeze(1) for f in cond_features]
        dec = self.dynamic_vq_model.decoder(dyn_h, cond_features, cond=self.cond_enable, attn_mask=attn_mask)
        if not return_dict:
            if return_loss:
                return (
                    dec['output'],
                    ref_dec['output'],
                    commit_loss,
                    dyn_commit_loss,
                    entropy_loss,
                    dyn_entropy_loss
                )
            return dec['output'], ref_dec['output']
        return CompressiveVQDecoderOutput(sample=dec['output'], ref_sample=ref_dec['output'], commit_loss=commit_loss, dyn_commit_loss=dyn_commit_loss, entropy_loss=entropy_loss, dyn_entropy_loss=dyn_entropy_loss)
def list_all_files(folder_path):
    path_img = []
    for root, dirs, files in os.walk(folder_path):
        files.sort()
        for file_name in files:
            path_img.append(os.path.join(root, file_name))

    return path_img




