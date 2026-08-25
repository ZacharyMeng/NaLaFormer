"""NaLaFormer: Norm-Aware Linear Attention Transformer for vision.

Official PyTorch implementation of
"Norm x Direction: Restoring the Missing Query Norm in Vision Linear
Attention" (ICML 2026).

NaLaFormer is a hierarchical vision backbone whose core is the
Norm-Aware Linear Attention (NaLa) module. It decomposes queries and
keys into a *norm* and a *direction* component:

  * a query-norm-aware feature map restores the spikiness of the
    attention distribution that vanilla linear attention loses through
    normalization;
  * a cosine-based direction similarity guarantees non-negativity
    without nullifying valid inner-product interactions.

Model variants (registered in timm):
    NALAFORMER_XT / NALAFORMER_T / NALAFORMER_S / NALAFORMER_B / NALAFORMER_L
"""
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from typing import Tuple


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class NormAwareLinearAttn(nn.Module):
    """Norm-Aware Linear Attention (NaLa).

    Given queries q and keys k, applies the Norm x Direction (ND)
    decomposition:

      1. split q into its L2 norm ||q|| and direction q / ||q||;
      2. inject ||q|| into the feature map via a tanh-gated power,
         restoring the query-norm -> attention-spikiness correlation;
      3. use a ReLU-split + cosine-style direction similarity that is
         non-negative yet preserves signed inner-product information.

    Linear complexity O(N) in the number of tokens N.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkvg = nn.Linear(dim, dim * 4)
        self.act = nn.SiLU()
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(-1, -2)
        N = H * W
        head_dim = self.head_dim
        num_heads = self.num_heads

        qkvg = self.qkvg(x).reshape(B, H * W, 4, C).permute(2, 0, 1, 3)
        qkvg = qkvg.float()
        q, k, v, g = qkvg[0], qkvg[1], qkvg[2], qkvg[3]

        lepe = self.lepe(v.reshape(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(B, N, -1)

        q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3).float()
        k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3).float()
        v = v.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3).float()

        k = k ** 3
        q1 = torch.relu(q)
        k1 = torch.relu(k)
        q2 = torch.relu(-q)
        k2 = torch.relu(-k)

        q_norm = q.norm(dim=-1, p=2, keepdim=True)

        q1 = q1 / q_norm
        q2 = q2 / q_norm
        power = 3 * (0.5 + torch.tanh(q_norm))
        q1 = q1 ** power
        q2 = q2 ** power

        q1_ = theta_shift(q1, sin, cos)
        k1_ = theta_shift(k1, sin, cos)
        q2_ = theta_shift(q2, sin, cos)
        k2_ = theta_shift(k2, sin, cos)

        q_ = torch.cat([q1_, q2_], dim=-1)
        k_ = torch.cat([k1_, k2_], dim=-1)
        q = torch.cat([q1, q2], dim=-1)
        k = torch.cat([k1, k2], dim=-1)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))
        x = q_ @ kv * z

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = x + lepe
        x = self.ln(x) * self.act(g)
        x = self.out_proj(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


class VanillaSelfAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkvg = nn.Linear(dim, dim * 4)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.out_proj = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(-1, -2)
        N = H * W
        head_dim = self.head_dim
        num_heads = self.num_heads

        qkvg = self.qkvg(x).reshape(B, H * W, 4, C).permute(2, 0, 1, 3)
        q, k, v, g = qkvg[0], qkvg[1], qkvg[2], qkvg[3]

        lepe = self.lepe(v.reshape(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(B, H * W, C)

        q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3).float()
        k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3).float()
        v = v.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3).float()

        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)

        attn = torch.softmax(self.scale * q @ k.transpose(-1, -2), dim=-1)
        res = attn @ v
        res = res.permute(0, 2, 1, 3).reshape(B, H * W, C)
        res = res + lepe.float()
        res = self.ln(res) * self.act(g)
        res = self.out_proj(res).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return res


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        subconv=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, 3, 1, 1, groups=ffn_dim) if subconv else None
        self.ffn_dim = ffn_dim

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.dwconv is not None:
            self.dwconv.reset_parameters()

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = x.contiguous()
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.dwconv is not None:
            x = x.reshape(B, H, W, self.ffn_dim).permute(0, 3, 1, 2)
            residual = x
            x = self.dwconv(x)
            x = x + residual
        x = x.reshape(B, self.ffn_dim, H * W).permute(0, 2, 1)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


class Block(nn.Module):

    def __init__(self, flag, embed_dim, num_heads, ffn_dim, drop_path=0., layerscale=False, layer_init_value=1e-6):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.norm1 = LayerNorm2d(embed_dim, eps=1e-6)
        assert flag in ['l', 'v']
        if flag == 'l':
            self.attn = NormAwareLinearAttn(embed_dim, num_heads)
        else:
            self.attn = VanillaSelfAttention(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = LayerNorm2d(embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)
        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_value * torch.ones(1, embed_dim, 1, 1), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_value * torch.ones(1, embed_dim, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        x = x + self.cpe1(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), sin, cos))
            x = x + self.cpe2(x)
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), sin, cos))
            x = x + self.cpe2(x)
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.reduction(x)
        x = self.norm(x)
        return x


class BasicLayer(nn.Module):

    def __init__(self, flags, embed_dim, out_dim, depth, num_heads,
                 ffn_dim=96., drop_path=0.,
                 downsample: PatchMerging = None,
                 layerscale=False, layer_init_value=1e-6):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.RoPE = RoPE(embed_dim, num_heads)

        self.blocks = nn.ModuleList([
            Block(flags[i], embed_dim, num_heads, ffn_dim,
                  drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_value)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor):
        _, _, h, w = x.size()
        sin, cos = self.RoPE((h, w))
        for blk in self.blocks:
            x = blk(x, sin, cos)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class RoPE(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    def forward(self, slen: Tuple[int]):
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :])
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :])
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1)
        sin = torch.cat([sin_h, sin_w], -1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :])
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :])
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1)
        cos = torch.cat([cos_h, cos_w], -1)
        return sin.flatten(0, 1), cos.flatten(0, 1)


class NALAFORMER(nn.Module):
    """NaLaFormer hierarchical vision backbone.

    Four stages of NaLa / vanilla-attention blocks with patch merging
    in between, followed by a projection + BN + linear classifier.

    Args:
        in_chans (int): number of input image channels.
        num_classes (int): number of classification classes.
        flagss (list[list[str]]): per-stage block types,
            'l' = NaLa linear attention, 'v' = vanilla softmax attention.
        embed_dims (list[int]): per-stage embedding dimensions.
        depths (list[int]): per-stage block counts.
        num_heads (list[int]): per-stage attention head counts.
        mlp_ratios (list[float]): per-stage FFN expansion ratios.
        drop_path_rate (float): maximum stochastic depth rate.
        projection (int): projection dimension before the classifier.
        layerscales (list[bool]): per-stage LayerScale switches.
        layer_init_values (list[float]): per-stage LayerScale init values.
    """

    def __init__(self, in_chans=3, num_classes=1000, flagss=[['l'] * 10, ['l'] * 10, ['v', 'v'] * 10, ['v'] * 10],
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1,
                 projection=1024, layerscales=[False, False, False, False], layer_init_values=[1e-6, 1e-6, 1e-6, 1e-6]):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                flags=flagss[i_layer],
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                layerscale=layerscales[i_layer],
                layer_init_value=layer_init_values[i_layer]
            )
            self.layers.append(layer)

        self.proj = nn.Linear(self.num_features, projection)
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Conv2d(projection, num_classes, 1) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.projection = projection

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except Exception:
                pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        x = self.proj(x)
        x = x.reshape(B, H, W, self.projection).permute(0, 3, 1, 2)
        x = self.norm(x)
        x = self.swish(x)
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x).flatten(1)
        return x


@register_model
def NALAFORMER_T(args=None):
    model = NALAFORMER(
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 6, 2],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        drop_path_rate=0.1,
        projection=1024,
        layerscales=[True, True, True, True],
        layer_init_values=[1, 1, 1, 1]
    )
    model.default_cfg = _cfg()
    return model


@register_model
def NALAFORMER_S(args=None):
    model = NALAFORMER(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 5, 9, 3],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[3.5, 3.5, 3.5, 3.5],
        drop_path_rate=0.15,
        projection=1024,
        layerscales=[True, True, True, True],
        layer_init_values=[1, 1, 1, 1]
    )
    model.default_cfg = _cfg()
    return model


@register_model
def NALAFORMER_B(args=None):
    model = NALAFORMER(
        embed_dims=[96, 192, 384, 512],
        depths=[4, 6, 12, 6],
        num_heads=[1, 2, 6, 8],
        mlp_ratios=[4, 4, 4, 4],
        drop_path_rate=0.4,
        projection=1024,
        layerscales=[True, True, True, True],
        layer_init_values=[1, 1, 1e-6, 1e-6]
    )
    model.default_cfg = _cfg()
    return model


@register_model
def NALAFORMER_L(args=None):
    model = NALAFORMER(
        embed_dims=[96, 192, 448, 640],
        depths=[4, 7, 19, 8],
        num_heads=[1, 2, 7, 10],
        mlp_ratios=[3.5, 3.5, 3.5, 3.5],
        drop_path_rate=0.55,
        projection=1024,
        layerscales=[True, True, True, True],
        layer_init_values=[1e-6, 1e-6, 1e-6, 1e-6]
    )
    model.default_cfg = _cfg()
    return model


@register_model
def NALAFORMER_XT(args=None):
    model = NALAFORMER(
        embed_dims=[32, 64, 128, 384],
        depths=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 12],
        mlp_ratios=[3.5, 3.5, 3.5, 3.5],
        drop_path_rate=0.1,
        projection=1024,
        layerscales=[True, True, True, True],
        layer_init_values=[1, 1, 1, 1]
    )
    model.default_cfg = _cfg()
    return model
