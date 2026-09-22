"""Tiny literal Torch reference for RAEv2 DDT conversion tests.

This mirrors nanovisionx/RAEv2's DDT.py/model_utils.py but replaces timm's
PatchEmbed with its equivalent Conv2d+flatten implementation.  It is a test
helper, not production model code.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


def _rotate_half(x):
    x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    return torch.stack((-x[..., 1], x[..., 0]), dim=-1).flatten(-2)


class RoPE(nn.Module):
    def __init__(self, dim, vis_len, cond_len=0, theta=10_000.0):
        super().__init__()
        half_dim, side = dim // 2, int(vis_len**0.5)
        frequencies = 1.0 / (
            theta ** (torch.arange(0, half_dim, 2).float() / half_dim)
        )
        base = torch.outer(torch.arange(side).float(), frequencies)
        angles = torch.cat(
            (
                base[:, None].expand(-1, side, -1),
                base[None].expand(side, -1, -1),
            ),
            dim=-1,
        ).reshape(vis_len, half_dim)
        angles = torch.cat((angles, torch.zeros(cond_len, half_dim)), dim=0)
        angles = angles.repeat_interleave(2, dim=-1)
        self.register_buffer("freqs_cos", angles.cos())
        self.register_buffer("freqs_sin", angles.sin())

    def forward(self, x):
        return x * self.freqs_cos + _rotate_half(x) * self.freqs_sin


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


class NormAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope, attn_mask=None):
        batch, length, dim = x.shape
        q = self.q(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = rope(self.q_norm(q)), rope(self.k_norm(k))
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        output = output.transpose(1, 2).reshape(batch, length, dim)
        return self.proj(output)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = NormAttention(dim, num_heads)
        self.mlp = SwiGLUFFN(dim, int(2 / 3 * dim * mlp_ratio))

    def forward(self, x, rope, attn_mask=None):
        x = x + self.attn(self.norm1(x), rope, attn_mask)
        return x + self.mlp(self.norm2(x))


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DecoderBlock(EncoderBlock):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__(dim, num_heads, mlp_ratio)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, condition, rope, attn_mask=None):
        modulation = self.adaln_modulation(condition)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = modulation.chunk(6, -1)
        x = x + gate_a * self.attn(
            _modulate(self.norm1(x), shift_a, scale_a), rope, attn_mask
        )
        return x + gate_m * self.mlp(_modulate(self.norm2(x), shift_m, scale_m))


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, channels):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * channels)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, condition):
        shift, scale = self.adaln_modulation(condition).chunk(2, -1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class GaussianFourierEmbedding(nn.Module):
    def __init__(self, dim, num_tokens, embedding_size=256):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.learnable_tokens = nn.Parameter(torch.randn(num_tokens, dim) / math.sqrt(dim))

    def forward(self, t, return_base_embed=False):
        angles = t[:, None] * self.W[None] * (2 * torch.pi)
        base = self.mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))
        if return_base_embed:
            base = base[:, None]
            return base, base + self.learnable_tokens
        return base[:, None] + self.learnable_tokens


class ConditionEmbedder(nn.Module):
    def __init__(self, dim, num_classes, num_tokens):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, dim)
        self.learnable_tokens = nn.Parameter(torch.randn(num_tokens, dim) / math.sqrt(dim))

    def forward(self, labels):
        return self.embedding_table(labels)[:, None] + self.learnable_tokens


class PatchEmbed(nn.Module):
    def __init__(self, input_size, patch_size, channels, dim):
        super().__init__()
        self.proj = nn.Conv2d(channels, dim, patch_size, stride=patch_size)
        self.num_patches = (input_size // patch_size) ** 2

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TinyOfficialRAEv2DDT(nn.Module):
    """Tiny square DiTwDDTHeadIG with the official state-dict layout."""

    def __init__(self):
        super().__init__()
        input_size, channels = 2, 3
        enc_dim, dec_dim = 16, 32
        enc_heads, dec_heads = 4, 8
        enc_depth, dec_depth = 2, 1
        mlp_ratio = 2.0
        self.in_channels = channels
        self.enc_hidden_size = enc_dim
        self.num_enc_blocks = enc_depth
        self.num_dec_blocks = dec_depth
        self.s_patch_size = self.x_patch_size = 1
        self.base_model_depth = 1
        self.s_embedder = PatchEmbed(input_size, 1, channels, enc_dim)
        self.x_embedder = PatchEmbed(input_size, 1, channels, dec_dim)
        self.s_projector = nn.Linear(enc_dim, dec_dim)
        self.t_embedder = GaussianFourierEmbedding(enc_dim, 4)
        self.ctx_embedder = ConditionEmbedder(enc_dim, 10, 8)
        blocks = [EncoderBlock(enc_dim, enc_heads, mlp_ratio) for _ in range(enc_depth)]
        blocks += [DecoderBlock(dec_dim, dec_heads, mlp_ratio) for _ in range(dec_depth)]
        self.blocks = nn.ModuleList(blocks)
        self.final_layer = FinalLayer(dec_dim, 1, channels)
        self.base_final_layer = FinalLayer(enc_dim, 1, channels)
        self.enc_rope = RoPE(enc_dim // enc_heads, 4, 12)
        self.dec_rope = RoPE(dec_dim // dec_heads, 4)
        self.cond_arch = SimpleNamespace(num_t_tokens=4, num_c_tokens=8)

    def unpatchify(self, x):
        batch, _, channels = x.shape
        return x.reshape(batch, 2, 2, 1, 1, channels).permute(0, 5, 1, 3, 2, 4).reshape(
            batch, channels, 2, 2
        )

    def forward(self, x, t, **condition_kwargs):
        s = self.s_embedder(x)
        t_base, t_tokens = self.t_embedder(t, return_base_embed=True)
        sequence = torch.cat((s, t_tokens, self.ctx_embedder(condition_kwargs["context"])), 1)
        base = None
        for index in range(self.num_enc_blocks):
            sequence = self.blocks[index](sequence, self.enc_rope)
            if index + 1 == self.base_model_depth:
                base = sequence[:, : self.s_embedder.num_patches]
        condition = self.s_projector(
            F.silu(t_base + sequence[:, : self.s_embedder.num_patches])
        )
        x_tokens = self.x_embedder(x)
        for index in range(self.num_dec_blocks):
            x_tokens = self.blocks[self.num_enc_blocks + index](
                x_tokens, condition, self.dec_rope
            )
        output = self.unpatchify(self.final_layer(x_tokens, condition))
        base = F.silu(t_base + base)
        base_output = self.unpatchify(self.base_final_layer(base, base))
        return output, base_output
