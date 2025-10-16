import torch
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

PADDING_MODE="zeros"

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad3d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad3d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad3d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

class GroupNormSpatial(nn.Module):
    """GroupNorm with spatial dimensions ignored."""
    def __init__(self, num_groups, num_channels, epsilon: float = 1e-5, affine=True):
        super().__init__()
        self.norm_fn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=epsilon, affine=affine)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 5:
            b, c, t, h, w = inputs.shape
            inputs = rearrange(inputs, "b c t h w -> (b t) c h w")
            out = self.norm_fn(inputs)
            return rearrange(out, "(b t) c h w -> b c t h w", b=b, t=t)
        else:
            out = self.norm_fn(inputs)


class BlurPool3D(nn.Module):
    def __init__(
        self,
        channels,
        pad_type="reflect",
        filt_size=3,
        stride=2,
        pad_off=0,
    ):
        super(BlurPool3D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        if self.filt_size == 1:
            a = np.array(
                [
                    1.0,
                ]
            )
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        filt_2d = a[:, None] * a[None, :]
        filt_3d = torch.Tensor(a[:, None, None] * filt_2d[None, :, :])

        filt = filt_3d / torch.sum(filt_3d)  # SCH: modified to it 3D
        
        self.register_buffer("filt", filt[None, None, :, :, :].repeat((self.channels, 1, 1, 1, 1)))

        # self.filt = filt[None, None, :, :, :].repeat((self.channels, 1, 1, 1, 1))
        # self.filt.requires_grad_(False)

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride, :: self.stride]
            else:
                return self.pad(inp)[:, :, :: self.stride, :: self.stride]
        else:
            padded_inp = self.pad(inp)
            return F.conv3d(padded_inp, self.filt.to(device=inp.device, dtype=inp.dtype), stride=self.stride, groups=inp.shape[1])


def init_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ResBlockDown(nn.Module):
    """3D StyleGAN ResBlock for D."""

    def __init__(
        self,
        in_channels,
        filters,
        activation_fn,
        num_groups=32,
    ):
        super().__init__()

        self.filters = filters
        self.activation_fn = activation_fn

        self.conv1 = nn.Conv3d(
            in_channels, in_channels, (3, 3, 3), padding="same", padding_mode=PADDING_MODE
        )
        self.norm1 = GroupNormSpatial(num_groups, in_channels)

        self.blur = BlurPool3D(in_channels, filt_size=3, stride=2)

        self.conv2 = nn.Conv3d(in_channels, self.filters, (1, 1, 1), bias=False, padding_mode=PADDING_MODE)
        self.conv3 = nn.Conv3d(
            in_channels, self.filters, (3, 3, 3), padding="same", padding_mode=PADDING_MODE)
        self.norm2 = GroupNormSpatial(num_groups, self.filters)
        


    def forward(self, x):
        residual = x
        x = self.conv1(x)
        # x = self.norm1(x)
        x = self.activation_fn(x)

        residual = self.blur(residual)

        residual = self.conv2(residual)


        x = self.blur(x)

        x = self.conv3(x)
        # x = self.norm2(x)
        x = self.activation_fn(x)
        out = (residual + x) / math.sqrt(2)
        return out


class LeCAM_EMA(object):
    def __init__(self, init=0., decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(logits_real).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(logits_fake).item() * (1 - self.decay)

def sigmoid_cross_entropy_with_logits(*, labels: torch.Tensor,
                                      logits: torch.Tensor) -> torch.Tensor:
    """Sigmoid cross entropy loss.

    We use a stable formulation that is equivalent to the one used in TensorFlow.
    The following derivation shows how we arrive at the formulation:

    .. math::
          z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
        = (1 - z) * x + log(1 + exp(-x))
        = x - x * z + log(1 + exp(-x))

    For x < 0, the following formula is more stable:
    .. math::
          x - x * z + log(1 + exp(-x))
        = log(exp(x)) - x * z + log(1 + exp(-x))
        = - x * z + log(1 + exp(x))

    We combine the two cases (x<0, x>=0) into one formula as follows:
    .. math::
        max(x, 0) - x * z + log(1 + exp(-abs(x)))

    This function is vmapped, so it is written for a single example, but can
    handle a batch of examples.

    Args:
      labels: The correct labels.
      logits: The output logits.

    Returns:
      The binary cross entropy loss for each given logit.
    """
    # The final formulation is: max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # To allow computing gradients at zero, we define custom versions of max and
    # abs functions just like in tf.nn.sigmoid_cross_entropy_with_logits.
    zeros = torch.zeros_like(logits, dtype=logits.dtype)
    condition = (logits >= zeros)
    relu_logits = torch.where(condition, logits, zeros)
    neg_abs_logits = torch.where(condition, -logits, logits)
    return relu_logits - logits * labels + torch.log1p(torch.exp(neg_abs_logits))
class StyleGANDiscriminator3D(nn.Module):
    """StyleGAN Discriminator."""
    def __init__(self, use_lecam_ema=False):
        super().__init__()
        self.spatial_size = 256
        self.frame_size = 4
        self.filters = 128
        self.activation_fn = nn.LeakyReLU(negative_slope=0.2)
        self.channel_multipliers = [2, 4]
        input_channels = [self.filters,] + [multiplier * self.filters for multiplier in self.channel_multipliers[:-1]]
        self.conv_in = nn.Conv3d(3, self.filters, kernel_size=(3, 3, 3), padding="same", padding_mode=PADDING_MODE)

        self.res_blocks = nn.Sequential(
            *[ResBlockDown(in_channels=input_channels[i], filters=self.channel_multipliers[i] * self.filters, activation_fn=self.activation_fn) for i in range(len(self.channel_multipliers))]
        )

        self.norm_fn = GroupNormSpatial(32, self.filters * self.channel_multipliers[-1])
        self.conv_out = nn.Conv3d(
            self.filters * self.channel_multipliers[-1], 
            self.filters * self.channel_multipliers[-1], 
            kernel_size=(3, 3, 3),
            padding="same",
            padding_mode=PADDING_MODE)
        self.use_lecam_ema = use_lecam_ema
        if self.use_lecam_ema:
            self.lecam_ema = LeCAM_EMA()


        time_downsample_factor = 2 ** len(self.channel_multipliers)
        spatial_downsample_factor = 2 ** len(self.channel_multipliers)
        assert self.spatial_size % spatial_downsample_factor == 0, \
        f"spatial_size: {self.spatial_size}, spatial_downsample_factor: {spatial_downsample_factor}"        
        # self.logit_input_feats = (self.spatial_size // spatial_downsample_factor) ** 2 * (self.frame_size // time_downsample_factor+1) * self.filters * self.channel_multipliers[-1]
        self.logit_input_feats = 3670016
        self.linear0 = nn.Linear(
            self.logit_input_feats,
            self.filters * self.channel_multipliers[-1]
        )
        
        self.linear1 = nn.Linear(
            self.filters * self.channel_multipliers[-1],                
            1
        )     
        self.apply(init_weights)


    def forward(self, x):
        x = self.conv_in(x)
        x = self.activation_fn(x)
        x = self.res_blocks(x)
        x = self.conv_out(x)
        x = self.norm_fn(x)
        x = self.activation_fn(x)
        x = x.view(x.shape[0], -1)
        assert self.logit_input_feats == x.shape[1]
        x = self.linear0(x)
        x = self.activation_fn(x)
        x = self.linear1(x)
        return x