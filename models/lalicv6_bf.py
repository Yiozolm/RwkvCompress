import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from einops import rearrange

from compressai.registry import register_model
from compressai.models import Elic2022Official
from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
    HyperpriorLatentCodec,
)
from compressai.layers import (
    CheckerboardMaskedConv2d,
    conv1x1,
    conv3x3,
    sequential_channel_ramp,
)



HEAD_SIZE = 16  # origin:64
T_MAX = 128 * 128  # for training on 256x256 crop


# T_MAX = 1024 * 1024  # for inference

current_file_dir = os.path.dirname(os.path.abspath(__file__))
wkv6_cuda = load(
        name="wkv6",
        sources=[
            os.path.join(current_file_dir, "cuda_v6_bf16/wkv6_op.cpp"),
            os.path.join(current_file_dir, "cuda_v6_bf16/wkv6_cuda.cu"),
        ],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--maxrregcount 60",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "-gencode arch=compute_86,code=sm_86",
            f"-D_N_={HEAD_SIZE}",
            f"-D_T_={T_MAX}",
        ],
    )



class BiWKV6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16,
                            memory_format=torch.contiguous_format)  #.uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C // H)
            return (None, None, None, None, gr, gk, gv, gw, gu)


def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return BiWKV6.apply(B, T, C, H, r, k, v, w, u)

class OmniShift(nn.Module):
    # Reparameterized 5x5 depth-wise convolution,
    # from RestoreRWKV, https://github.com/Yaziwel/Restore-RWKV

    def __init__(self, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.conv5x5 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)

        out = (
                self.alpha[0] * x
                + self.alpha[1] * out1x1
                + self.alpha[2] * out3x3
                + self.alpha[3] * out5x5
        )
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution

        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))

        combined_weight = (
                self.alpha[0] * identity_weight
                + self.alpha[1] * padded_weight_1x1
                + self.alpha[2] * padded_weight_3x3
                + self.alpha[3] * self.conv5x5.weight
        )
        device = self.conv5x5_reparam.weight.device
        combined_weight = combined_weight.to(device)
        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)

    def forward(self, x):
        if self.training:
            self.repram_flag = True
            out = self.forward_train(x)
        elif self.training is False and self.repram_flag is True:
            self.reparam_5x5()
            self.repram_flag = False
            out = self.conv5x5_reparam(x)
        elif self.training is False and self.repram_flag is False:
            out = self.conv5x5_reparam(x)

        return out


class SpatialMix_BiV6(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        attn_dim = dim

        self.head_size = HEAD_SIZE
        self.n_head = self.dim // self.head_size
        assert self.dim == self.head_size * self.n_head, f'Total dim:{self.dim},n_head:{self.n_head},head_size:{self.head_size},rectify your HEADSIZE'
        self.device = None

        self.shift = OmniShift(dim=dim) # dim = n_embd, attn_dim = attn_sz
        self.key = nn.Linear(dim, attn_dim, bias=False)
        self.value = nn.Linear(dim, attn_dim, bias=False)
        self.receptance = nn.Linear(dim, attn_dim, bias=False)
        self.gate = nn.Linear(dim, attn_dim, bias=False)

        self.output = nn.Linear(attn_dim, dim, bias=False)

        self.ln_x = nn.GroupNorm(self.n_head, attn_dim, eps=1e-5)

        # vrwkv in restore-rwkv
        with torch.no_grad():
            # ddd = torch.ones(1, 1, self.dim)

            # fancy time_mix
            self.time_maa_x = nn.Parameter(torch.randn(1, 1, self.dim))
            self.time_maa_w = nn.Parameter(torch.randn(1, 1, self.dim))
            self.time_maa_k = nn.Parameter(torch.randn(1, 1, self.dim))
            self.time_maa_v = nn.Parameter(torch.randn(1, 1, self.dim))
            self.time_maa_r = nn.Parameter(torch.randn(1, 1, self.dim))
            self.time_maa_g = nn.Parameter(torch.randn(1, 1, self.dim))

            TIME_MIX_EXTRA_DIM = 32  # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(self.dim, TIME_MIX_EXTRA_DIM * 5).uniform_(-1e-4, 1e-4))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, self.dim).uniform_(-1e-4, 1e-4))

            # fancy time_decay
            self.time_decay1 = nn.Parameter(torch.randn(1, 1, attn_dim))
            self.time_decay2 = nn.Parameter(torch.randn(1, 1, attn_dim))

            TIME_DECAY_EXTRA_DIM = 64
            self.time_decay_w1_1 = nn.Parameter(torch.zeros(self.dim, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
            self.time_decay_w1_2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, attn_dim).uniform_(-1e-4, 1e-4))
            self.time_faaaa_1 = nn.Parameter(torch.randn(self.n_head, self.head_size))

            self.time_decay_w2_1 = nn.Parameter(torch.zeros(self.dim, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
            self.time_decay_w2_2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, attn_dim).uniform_(-1e-4, 1e-4))
            self.time_faaaa_2 = nn.Parameter(torch.randn(self.n_head, self.head_size))

    def jit_func(self, x, resolution):
        B, T, C = x.size()
        H, W = resolution
        xx = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)
        xx = self.shift(xx)
        xx = rearrange(xx, "B C H W -> B (H W) C")

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)

        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        g = F.silu(self.gate(xg))

        ww1 = torch.tanh(xw @ self.time_decay_w1_1) @ self.time_decay_w1_2  # [B, T, C]
        w1 = self.time_decay1 + ww1

        ww2 = torch.tanh(xw @ self.time_decay_w2_1) @ self.time_decay_w2_2  # [B, T, C]
        w2 = self.time_decay2 + ww2

        return r, k, v, g, w1, w2

    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        r, k, v, g, w1, w2 = self.jit_func(x, resolution)

        v = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w1, u=self.time_faaaa_1)

        H, W = resolution

        r = rearrange(r, 'B (H W) C -> B (W H) C', H=H, W=W)
        k = rearrange(k, 'B (H W) C -> B (W H) C', H=H, W=W)
        v = rearrange(v, 'B (H W) C -> B (W H) C', H=H, W=W)

        v = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w2, u=self.time_faaaa_2)
        x = rearrange(v, 'B (W H) C -> B (H W) C', H=H, W=W)

        return self.jit_func_2(x, g)


class ChannelMix_V6(nn.Module):
    def __init__(self, dim, hidden_rate=4,
                 key_norm=False):
        super().__init__()
        self.n_embd = dim
        hidden_dim = int(hidden_rate * dim)

        self.shift = OmniShift(dim=dim)
        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(hidden_dim, dim, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_dim)
        else:
            self.key_norm = None

    def forward(self, x, resolution):
        H, W = resolution
        x = rearrange(x, 'B (H W) C -> B C H W', H=H, W=W)
        x = self.shift(x)
        x = rearrange(x, 'B C H W -> B (H W) C')

        k = self.key(x)
        k = torch.square(torch.relu(k))

        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv

        return x

class RwkvBlock_BiV6(nn.Module):
    def __init__(self, dim, hidden_rate=4, with_ckpt=False):
        super().__init__()
        self.with_ckpt = with_ckpt

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.att = SpatialMix_BiV6(dim)
        self.ffn = ChannelMix_V6(dim, hidden_rate)
        self.gamma1 = nn.Parameter(torch.ones((dim)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((dim)), requires_grad=True)

    def _forward(self, x):
        B, C, H, W = x.shape
        resolution = (H, W)

        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.gamma1 * self.att(self.ln1(x), resolution)
        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x

    def forward(self, x):
        if self.with_ckpt and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False
            )
        else:
            return self._forward(x)

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


def form_modules(*modules):
    flattened = []
    for m in modules:
        if isinstance(m, list):
            flattened.extend(m)
        else:
            flattened.append(m)
    return nn.Sequential(*flattened)


class EntropyParametersBlock(nn.Module):
    def __init__(self, dim, out_dim, expansion_factor=2, **kwargs):
        super().__init__()

        hidden_dim = int(expansion_factor * out_dim)
        self.mix = nn.Conv2d(dim, out_dim, 1)
        self.norm = nn.LayerNorm(out_dim)
        self.key = nn.Linear(out_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, out_dim, bias=False)
        self.receptance = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.mix(x)
        identity = x
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        k = self.key(x)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x + identity

@register_model("LALICv6")
class LALICv6(Elic2022Official):
    def __init__(
            self,
            N=128,
            M=320,
            dims=[96, 144, 256, 320, 256, 192],
            depths=[2, 4, 6, 6],
            groups=None,
            use_ckpt=False,
            **kwargs,
    ):
        super().__init__(N=N, M=M, groups=groups, **kwargs)
        # self.N = N
        # self.M = M
        N1, N2, N3, N4, N5, N6 = dims
        L1, L2, L3, L4 = depths
        M = N4

        # flatten the list
        self.g_a = form_modules(
            conv(3, N1, kernel_size=5),
            [RwkvBlock_BiV6(N1) for _ in range(L1)],
            conv(N1, N2, kernel_size=3),
            [RwkvBlock_BiV6(N2) for i in range(L2)],
            conv(N2, N3, kernel_size=3),
            [RwkvBlock_BiV6(N3) for _ in range(L3)],
            conv(N3, N4, kernel_size=3),
        )

        self.g_s = form_modules(
            deconv(N4, N3, kernel_size=3),
            [RwkvBlock_BiV6(N3) for _ in range(L3)],
            deconv(N3, N2, kernel_size=3),
            [RwkvBlock_BiV6(N2) for _ in range(L2)],
            deconv(N2, N1, kernel_size=3),
            [RwkvBlock_BiV6(N1) for _ in range(L1)],
            deconv(N1, 3, kernel_size=5),
        )

        self.h_a = form_modules(
            conv(N4, N5, kernel_size=5),
            [RwkvBlock_BiV6(N5) for _ in range(L4)],
            conv(N5, N6, kernel_size=5),
        )

        self.h_s = form_modules(
            deconv(N6, N5, kernel_size=5),
            [RwkvBlock_BiV6(N5) for _ in range(L4)],
            deconv(N5, N4, kernel_size=5),
        )

        # In [He2022], this is labeled "g_ch^(k)".
        channel_context = {
            f"y{k}": nn.Sequential(
                conv3x3(sum(self.groups[:k]), M),
                RwkvBlock_BiV6(M, hidden_rate=8),
                RwkvBlock_BiV6(M, hidden_rate=8),
                conv1x1(M, self.groups[k] * 2),
            )
            for k in range(1, len(self.groups))
        }

        # In [He2022], this is labeled "g_sp^(k)". Same as ELIC
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + M,
                self.groups[k] * 2,
                min_ch=N * 2,
                num_layers=3,
                make_layer=EntropyParametersBlock,  # two differences
                make_act=nn.Identity,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # [He2022] uses a "hyperprior" architecture, which reconstructs y using z.
        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                # Channel groups with space-channel context model (SCCTX):
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                # Side information branch containing z:
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N6),
                    h_a=self.h_a,
                    h_s=self.h_s,
                    quantizer="ste",
                ),
            },
        )

    @classmethod
    def from_state_dict(cls, state_dict, strict=True):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict, strict=strict)
        return net
