#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESPERIMENTO: DiT DDPM su CIFAR-10
===================================
Diffusion Transformer (DiT) con DDPM su CIFAR-10.

Motivazione: Il risultato U-Net (NOVA 0.0382 > GELU 0.0372, GELU vince) è
spiegato come "Smoothness Trade-off": la U-Net usa BatchNorm, che entra in
conflitto con la non-monotonia di NOVA. Il DiT (Peebles & Xie, 2023) usa
LayerNorm + AdaLN — esattamente il contesto dove NOVA eccelle nei ViT.
L'ipotesi è che NOVA recuperi quando il backbone generativo è un Transformer.

Architettura:
  Input: x_noisy (32×32×3) + timestep t
  ├── Patchify: Conv2d 4×4 stride 4 → 8×8 = 64 patch, proiezione a dim d
  ├── + Positional Embedding (learnable, 64 token)
  ├── N × DiT Block:
  │   ├── AdaLN(LayerNorm, condizionato su t)
  │   ├── Multi-Head Self-Attention
  │   ├── AdaLN(LayerNorm, condizionato su t)
  │   └── MLP: Linear(d, 4d) → ATTIVAZIONE → Linear(4d, d)
  ├── Final AdaLN → Linear(d, patch_size² × 3) = Linear(d, 48)
  └── Unpatchify → 32×32×3 (predizione del rumore ε)

Scaling:
  - Tiny (4L, 192d, 3h, ~1.5M params)
  - Small (6L, 256d, 4h, ~4M params)
  - Base (8L, 384d, 6h, ~10M params)

Uso:
    # Lancia tutto (tutte le scale, NOVA vs GELU su 2 GPU):
    python dit_cifar10.py

    # Singolo esperimento:
    python dit_cifar10.py --scale tiny --activation nova --gpu 0

    # Solo generazione plot (dopo aver completato gli esperimenti):
    python dit_cifar10.py --plot-only
"""

import argparse
import subprocess
import sys
import os
import time
import json
import glob
import math
import random
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as transforms

# ==============================================================
# CONFIGURAZIONE ESPERIMENTO
# ==============================================================
SEED = 42
DATASET = "CIFAR-10"
HARDWARE = "NVIDIA T4"
NUM_WORKERS = 4
PATCH_SIZE = 4
MLP_RATIO = 4
IMG_SIZE = 32
IMG_CHANNELS = 3
NUM_CLASSES = 10  # per class-conditional (non usato ora, unconditional)
T_DIFFUSION = 1000
EMA_DECAY = 0.9999
MAX_GRAD_NORM = 1.0
FID_NUM_SAMPLES = 10000
FID_EVERY = 10  # ogni quante epoche calcolare FID/IS
SAMPLE_EVERY = 10  # ogni quante epoche generare campioni visivi
SAMPLE_GRID = 8  # griglia 8×8

SCALING_CONFIGS = {
    "tiny": {
        "embed_dim": 192,
        "num_heads": 3,
        "num_layers": 4,
        "batch_size": 512,
        "lr": 3e-4,
        "epochs": 100,
        "warmup_epochs": 5,
        "weight_decay": 0.0,
    },
    "small": {
        "embed_dim": 256,
        "num_heads": 4,
        "num_layers": 6,
        "batch_size": 256,
        "lr": 3e-4,
        "epochs": 100,
        "warmup_epochs": 5,
        "weight_decay": 0.0,
    },
    "base": {
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 8,
        "batch_size": 128,
        "lr": 3e-4,
        "epochs": 100,
        "warmup_epochs": 5,
        "weight_decay": 0.0,
    },
}

ALL_ACTIVATIONS = ["nova", "gelu"]


# ==============================================================
# UTILITY
# ==============================================================
def _is_kaggle():
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def _get_results_dir():
    if _is_kaggle():
        d = "/kaggle/working/results"
    else:
        script_dir = os.path.abspath(os.path.dirname(__file__))
        d = os.path.join(script_dir, "..", "results")
    os.makedirs(d, exist_ok=True)
    return os.path.abspath(d)


def _get_data_root():
    if _is_kaggle():
        return "/kaggle/working/data"
    script_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(script_dir, "..", "data")


# ==============================================================
# RIPRODUCIBILITA'
# ==============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================
# NOVA: KERNEL CUDA + FALLBACK PYTHON
# ==============================================================

_nova_cuda_ext = None


def _compile_nova_cuda():
    from torch.utils.cpp_extension import load_inline

    cpp_source = """
    torch::Tensor nova_cuda_forward(torch::Tensor x, float beta);
    std::vector<torch::Tensor> nova_cuda_backward(
        torch::Tensor grad_output, torch::Tensor x, float beta);
    """

    cuda_source = r"""
    #include <torch/extension.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <vector>

    template <typename scalar_t>
    __global__ void nova_cuda_forward_kernel(
        const scalar_t* __restrict__ x,
        scalar_t* __restrict__ out,
        const float beta,
        const int size)
    {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            const float val = static_cast<float>(x[index]);
            const float bx = beta * val;
            const float sig = 1.0f / (1.0f + expf(-bx));
            const float gating = val * sig;
            const float rational = val / (1.0f + bx * bx);
            out[index] = static_cast<scalar_t>(gating - rational);
        }
    }

    template <typename scalar_t>
    __global__ void nova_cuda_backward_kernel(
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ x,
        scalar_t* __restrict__ grad_input,
        scalar_t* __restrict__ grad_beta_elem,
        const float beta,
        const int size)
    {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            const float go = static_cast<float>(grad_output[index]);
            const float val = static_cast<float>(x[index]);
            const float bx = beta * val;
            const float sig = 1.0f / (1.0f + expf(-bx));
            const float sig_deriv = sig * (1.0f - sig);
            const float d_gating_dx = sig + bx * sig_deriv;
            const float bx_sq = bx * bx;
            const float denom = 1.0f + bx_sq;
            const float d_rational_dx = (1.0f - bx_sq) / (denom * denom);
            grad_input[index] = static_cast<scalar_t>(
                go * (d_gating_dx - d_rational_dx));
            const float val_sq = val * val;
            const float d_gating_dbeta = val_sq * sig_deriv;
            const float d_rational_dbeta = -2.0f * beta * val_sq * val / (denom * denom);
            grad_beta_elem[index] = static_cast<scalar_t>(
                go * (d_gating_dbeta - d_rational_dbeta));
        }
    }

    torch::Tensor nova_cuda_forward(torch::Tensor x, float beta) {
        auto out = torch::empty_like(x);
        const int threads = 256;
        const int blocks = (x.numel() + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            x.scalar_type(), "nova_forward_cuda", ([&] {
                nova_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                    x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                    beta, x.numel());
            }));
        return out;
    }

    std::vector<torch::Tensor> nova_cuda_backward(
            torch::Tensor grad_output, torch::Tensor x, float beta) {
        auto grad_input = torch::empty_like(x);
        auto grad_beta_elem = torch::empty_like(x);
        const int threads = 256;
        const int blocks = (x.numel() + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            x.scalar_type(), "nova_backward_cuda", ([&] {
                nova_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                    grad_output.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
                    grad_input.data_ptr<scalar_t>(),
                    grad_beta_elem.data_ptr<scalar_t>(),
                    beta, x.numel());
            }));
        auto grad_beta = grad_beta_elem.sum();
        return {grad_input, grad_beta};
    }
    """

    return load_inline(
        name='nova_cuda_ext_dit',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['nova_cuda_forward', 'nova_cuda_backward'],
        with_cuda=True,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
    )


class _NOVAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta, nova_ext):
        x = x.contiguous()
        beta_val = beta.item()
        ctx.save_for_backward(x)
        ctx.beta_val = beta_val
        ctx.nova_ext = nova_ext
        return nova_ext.nova_cuda_forward(x, beta_val)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x, grad_beta = ctx.nova_ext.nova_cuda_backward(
            grad_output, x, ctx.beta_val)
        return grad_x, grad_beta, None


class NOVACuda(nn.Module):
    def __init__(self, nova_ext, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.nova_ext = nova_ext

    def forward(self, x):
        return _NOVAFunction.apply(x, self.beta, self.nova_ext)


class NOVAPython(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def forward(self, x):
        bx = self.beta * x
        return x * torch.sigmoid(bx) - x / (1.0 + bx ** 2)


def make_nova(beta=1.0):
    global _nova_cuda_ext
    if _nova_cuda_ext is None and torch.cuda.is_available():
        try:
            print("[NOVA] Compilazione kernel CUDA (30-60s alla prima esecuzione)...")
            _nova_cuda_ext = _compile_nova_cuda()
            print("[NOVA] Kernel CUDA compilato con successo.")
        except Exception as e:
            print(f"[NOVA] Compilazione CUDA fallita: {e}")
            print("[NOVA] Uso implementazione Python pura.")
    if _nova_cuda_ext is not None:
        return NOVACuda(_nova_cuda_ext, beta=beta), "cuda"
    return NOVAPython(beta=beta), "python"


def make_activation(name, beta=1.0):
    """Factory per creare l'attivazione. Ritorna (module, backend_info)."""
    if name == "nova":
        return make_nova(beta=beta)
    elif name == "gelu":
        return nn.GELU(), "builtin"
    elif name == "silu":
        return nn.SiLU(), "builtin"
    elif name == "mish":
        return nn.Mish(), "builtin"
    elif name == "relu":
        return nn.ReLU(), "builtin"
    else:
        raise ValueError(f"Attivazione non supportata: {name}")


# ==============================================================
# COSINE NOISE SCHEDULE (Nichol & Dhariwal, 2021)
# ==============================================================

class CosineNoiseSchedule:
    """Cosine schedule per DDPM (Improved DDPM, Nichol & Dhariwal 2021)."""

    def __init__(self, T=1000, s=0.008):
        self.T = T
        steps = T + 1
        t = torch.linspace(0, T, steps, dtype=torch.float64)
        f_t = torch.cos(((t / T) + s) / (1 + s) * (math.pi / 2)) ** 2
        alphas_cumprod = f_t / f_t[0]

        # β_t = 1 - ᾱ_t / ᾱ_{t-1}, clipped
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0, 0.999).float()

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Coefficienti utili
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Coefficienti per il posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def to(self, device):
        """Sposta tutti i tensori su device."""
        for attr in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                      'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                      'sqrt_recip_alphas', 'posterior_variance']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(self, x_0, t, noise=None):
        """Forward process: q(x_t | x_0) = N(sqrt(ᾱ_t)*x_0, (1-ᾱ_t)*I)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise


# ==============================================================
# EMA (Exponential Moving Average)
# ==============================================================

class EMA:
    """Exponential Moving Average dei parametri del modello."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        """Applica i pesi EMA al modello (salva backup)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Ripristina i pesi originali dal backup."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ==============================================================
# DiT ARCHITECTURE
# ==============================================================

class SinusoidalTimestepEmbedding(nn.Module):
    """Embedding sinusoidale per il timestep (Vaswani et al.)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimestepEmbedder(nn.Module):
    """Timestep → conditioning vector: sinusoidal → Linear → SiLU → Linear."""

    def __init__(self, embed_dim):
        super().__init__()
        self.sinusoidal = SinusoidalTimestepEmbedding(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),  # SiLU fisso (standard DiT)
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t):
        return self.mlp(self.sinusoidal(t))


class DiTBlock(nn.Module):
    """
    DiT Block con AdaLN (Adaptive Layer Normalization).

    Schema:
      c = conditioning (da timestep embedder)
      (γ1, β1, α1, γ2, β2, α2) = adaLN_modulation(SiLU(c))
      x = x + α1 * Attention(γ1 * LayerNorm(x) + β1)
      x = x + α2 * MLP(γ2 * LayerNorm(x) + β2)
    """

    def __init__(self, dim, num_heads, mlp_ratio=4, act_layer=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(dim * mlp_ratio)
        if act_layer is None:
            act_layer = nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            act_layer,
            nn.Linear(mlp_hidden, dim),
        )

        # AdaLN modulation: produce 6 parametri (γ1, β1, α1, γ2, β2, α2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

    def forward(self, x, c):
        """
        x: (B, N, D) — patch tokens
        c: (B, D) — conditioning vector da timestep
        """
        mod = self.adaLN_modulation(c).unsqueeze(1)  # (B, 1, 6*D)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # Attention branch
        h = self.norm1(x)
        h = h * (1 + gamma1) + beta1
        x = x + alpha1 * self.attn(h)

        # MLP branch
        h = self.norm2(x)
        h = h * (1 + gamma2) + beta2
        x = x + alpha2 * self.mlp(h)

        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class FinalLayer(nn.Module):
    """Layer finale del DiT: AdaLN → Linear → Unpatchify."""

    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)

    def forward(self, x, c):
        mod = self.adaLN_modulation(c).unsqueeze(1)  # (B, 1, 2*D)
        gamma, beta = mod.chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + gamma) + beta
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) per generazione di immagini.

    Patchify → PosEmbed → N×DiTBlock → FinalLayer → Unpatchify
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=192, num_heads=3, num_layers=4,
                 mlp_ratio=4, activation_name="gelu", activation_beta=1.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2  # 8×8 = 64

        # Patchify: Conv2d 4×4 stride 4
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size)

        # Positional embedding (learnable, 64 token)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim))

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(embed_dim)

        # DiT Blocks — ogni blocco ha la propria istanza di attivazione
        self.blocks = nn.ModuleList()
        self._activation_name = activation_name
        for _ in range(num_layers):
            act, _ = make_activation(activation_name, beta=activation_beta)
            block = DiTBlock(embed_dim, num_heads, mlp_ratio, act_layer=act)
            self.blocks.append(block)

        # Final layer
        self.final_layer = FinalLayer(embed_dim, patch_size, in_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Inizializzazione standard per Transformer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        # Zero-init per gli output di adaLN_modulation (DiT paper)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x):
        """(B, N, patch_size²×C) → (B, C, H, W)"""
        p = self.patch_size
        h = w = self.img_size // p
        c = self.in_channels
        x = x.reshape(-1, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        x = x.reshape(-1, c, h * p, w * p)
        return x

    def forward(self, x, t):
        """
        x: (B, C, H, W) — immagine rumorosa
        t: (B,) — timestep (long)
        Ritorna: (B, C, H, W) — predizione del rumore ε
        """
        # Patchify
        x = self.patch_embed(x)  # (B, D, h, w)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # Positional embedding
        x = x + self.pos_embed

        # Timestep conditioning
        c = self.t_embedder(t)  # (B, D)

        # DiT Blocks
        for block in self.blocks:
            x = block(x, c)

        # Final layer + unpatchify
        x = self.final_layer(x, c)  # (B, N, patch_size²×C)
        x = self.unpatchify(x)  # (B, C, H, W)

        return x


# ==============================================================
# DATI: CIFAR-10 normalizzato a [-1, 1]
# ==============================================================

def get_cifar10_loaders(batch_size, num_workers):
    """CIFAR-10 normalizzato a [-1, 1] per diffusion."""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data_root = _get_data_root()
    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


# ==============================================================
# FID / IS COMPUTATION (Inception v3)
# ==============================================================

@torch.no_grad()
def get_inception_features(images_tensor, device, batch_size=64):
    """
    Calcola features Inception v3 (2048-dim) e logits (1000-dim).

    images_tensor: (N, 3, 32, 32) in [-1, 1]
    Ritorna: features (N, 2048), probs (N, 1000)
    """
    # Carica Inception v3
    inception = torchvision.models.inception_v3(
        weights='IMAGENET1K_V1')
    # Salva i pesi del fc per calcolare logits manualmente
    fc_weight = inception.fc.weight.data.clone().to(device)
    fc_bias = inception.fc.bias.data.clone().to(device)
    # Rimpiazza fc con Identity per ottenere features pool (2048)
    inception.fc = nn.Identity()
    inception = inception.to(device).eval()

    # Preprocessing: [-1,1] → [0,1] → resize 299 → ImageNet normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    all_features = []
    all_probs = []

    N = images_tensor.shape[0]
    for i in range(0, N, batch_size):
        batch = images_tensor[i:i+batch_size].to(device)
        # [-1, 1] → [0, 1]
        batch = (batch + 1.0) / 2.0
        batch = batch.clamp(0, 1)
        # Resize a 299×299
        batch = F.interpolate(batch, size=(299, 299),
                              mode='bilinear', align_corners=False)
        # ImageNet normalize
        batch = (batch - mean) / std

        features = inception(batch)  # (B, 2048) grazie a fc=Identity
        logits = F.linear(features, fc_weight, fc_bias)  # (B, 1000)
        probs = F.softmax(logits, dim=1)

        all_features.append(features.cpu())
        all_probs.append(probs.cpu())

    del inception, fc_weight, fc_bias
    torch.cuda.empty_cache()

    return torch.cat(all_features, dim=0), torch.cat(all_probs, dim=0)


def compute_fid(mu1, sigma1, mu2, sigma2):
    """Calcola Fréchet Inception Distance."""
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return fid


def compute_inception_score(probs, splits=10):
    """Calcola Inception Score (IS) da probabilità softmax."""
    N = probs.shape[0]
    split_size = N // splits
    scores = []
    for k in range(splits):
        part = probs[k * split_size:(k + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))
    return float(np.mean(scores)), float(np.std(scores))


def compute_real_stats(data_loader, device, num_samples=10000):
    """Calcola mu e sigma delle features Inception v3 sui dati reali."""
    print("[FID] Calcolo statistiche dati reali...")
    images = []
    count = 0
    for imgs, _ in data_loader:
        images.append(imgs)
        count += imgs.shape[0]
        if count >= num_samples:
            break
    images = torch.cat(images, dim=0)[:num_samples]

    features, _ = get_inception_features(images, device)
    features_np = features.numpy()
    mu = np.mean(features_np, axis=0)
    sigma = np.cov(features_np, rowvar=False)
    print(f"[FID] Statistiche reali calcolate su {features_np.shape[0]} immagini.")
    return mu, sigma


# ==============================================================
# DDPM SAMPLING
# ==============================================================

@torch.no_grad()
def ddpm_sample(model, noise_schedule, shape, device):
    """
    Genera campioni con DDPM sampling.

    shape: (B, C, H, W)
    Ritorna: (B, C, H, W) in [-1, 1]
    """
    B = shape[0]
    x = torch.randn(shape, device=device)

    for t_idx in reversed(range(noise_schedule.T)):
        t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)

        # Predizione del rumore
        with autocast(device_type='cuda', dtype=torch.float16):
            eps_pred = model(x, t_batch)
        eps_pred = eps_pred.float()

        # Coefficienti
        alpha_t = noise_schedule.alphas[t_idx]
        alpha_bar_t = noise_schedule.alphas_cumprod[t_idx]
        beta_t = noise_schedule.betas[t_idx]

        # Mean del posterior
        mu = (1.0 / alpha_t.sqrt()) * (
            x - (beta_t / (1.0 - alpha_bar_t).sqrt()) * eps_pred
        )

        if t_idx > 0:
            sigma = beta_t.sqrt()
            z = torch.randn_like(x)
            x = mu + sigma * z
        else:
            x = mu

    return x.clamp(-1, 1)


@torch.no_grad()
def generate_samples_batched(model, noise_schedule, num_samples, device,
                             batch_size=256):
    """Genera num_samples campioni in batch."""
    all_samples = []
    remaining = num_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        shape = (bs, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
        samples = ddpm_sample(model, noise_schedule, shape, device)
        all_samples.append(samples.cpu())
        remaining -= bs
    return torch.cat(all_samples, dim=0)


# ==============================================================
# TRAINING & EVALUATION
# ==============================================================

def train_one_epoch(model, loader, noise_schedule, optimizer, scaler, device):
    """Training: epsilon-prediction con MSE loss."""
    model.train()
    running_loss = 0.0
    total = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        B = images.shape[0]

        # Sample random timesteps
        t = torch.randint(0, noise_schedule.T, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(images)

        # Forward process: q(x_t | x_0)
        x_t = noise_schedule.q_sample(images, t, noise=noise)

        # Predict noise
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=torch.float16):
            eps_pred = model(x_t, t)
            loss = F.mse_loss(eps_pred, noise)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * B
        total += B

    return running_loss / total


@torch.no_grad()
def evaluate(model, loader, noise_schedule, device):
    """Validation loss: MSE sui dati di test (con EMA weights)."""
    model.eval()
    running_loss = 0.0
    total = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        B = images.shape[0]

        t = torch.randint(0, noise_schedule.T, (B,), device=device)
        noise = torch.randn_like(images)
        x_t = noise_schedule.q_sample(images, t, noise=noise)

        with autocast(device_type='cuda', dtype=torch.float16):
            eps_pred = model(x_t, t)
            loss = F.mse_loss(eps_pred, noise)

        running_loss += loss.item() * B
        total += B

    return running_loss / total


def build_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-4, end_factor=1.0,
        total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ==============================================================
# ESPERIMENTO PRINCIPALE
# ==============================================================

def run_experiment(scale: str, activation: str, gpu: int) -> None:
    set_seed(SEED)

    cfg = SCALING_CONFIGS[scale]
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    embed_dim = cfg["embed_dim"]
    num_heads = cfg["num_heads"]
    num_layers = cfg["num_layers"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    epochs = cfg["epochs"]
    warmup_epochs = cfg["warmup_epochs"]
    weight_decay = cfg["weight_decay"]

    experiment_name = f"dit_cifar10_{scale}_{activation}"
    model_desc = (f"DiT-{scale.capitalize()} ({num_layers}L, {embed_dim}d, "
                  f"{num_heads}h, MLP×{MLP_RATIO}) + {activation.upper()}")

    print(f"\n{'='*60}")
    print(f"  ESPERIMENTO: {experiment_name}")
    print(f"  GPU: {gpu} ({torch.cuda.get_device_name(device)})")
    print(f"  Scala: {scale.upper()} | Attivazione: {activation.upper()}")
    print(f"  Architettura: {num_layers}L, {embed_dim}d, {num_heads}h")
    print(f"  Epoche: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Diffusion: T={T_DIFFUSION}, Cosine schedule, EMA={EMA_DECAY}")
    print(f"  Mixed Precision: FP16 (calcoli critici in FP32)")
    print(f"  Seed: {SEED}")
    print(f"{'='*60}\n")

    # --- Noise schedule ---
    noise_schedule = CosineNoiseSchedule(T=T_DIFFUSION)
    noise_schedule.to(device)

    # --- Modello ---
    model = DiT(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=IMG_CHANNELS,
        embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
        mlp_ratio=MLP_RATIO, activation_name=activation, activation_beta=1.0,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parametri totali: {num_params:,}")

    # --- EMA ---
    ema = EMA(model, decay=EMA_DECAY)

    # --- Dati ---
    train_loader, test_loader = get_cifar10_loaders(batch_size, NUM_WORKERS)

    # --- Ottimizzatore, Scheduler, Scaler ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, warmup_epochs, epochs)
    scaler = GradScaler()

    # --- Log setup ---
    results_dir = _get_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(results_dir, f"{experiment_name}_{timestamp}.json")

    log_data = {
        "esperimento": experiment_name,
        "data": datetime.now().strftime("%Y-%m-%d"),
        "hardware": f"{HARDWARE} (GPU {gpu})",
        "seed": SEED,
        "dataset": DATASET,
        "modello": model_desc,
        "obiettivo": (f"DiT DDPM {scale}: {activation} — ε-prediction, "
                      f"cosine schedule, {epochs} epoche"),
        "configurazione": {
            "scale": scale,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "patch_size": PATCH_SIZE,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "mlp_ratio": MLP_RATIO,
            "T_diffusion": T_DIFFUSION,
            "noise_schedule": "cosine",
            "ema_decay": EMA_DECAY,
            "grad_clip": MAX_GRAD_NORM,
            "mixed_precision": "fp16",
            "num_params": num_params,
        },
        "metriche": {},
        "epoche_log": [],
        "fid_log": [],
        "tempo_totale_sec": None,
    }

    # --- Pre-compute real stats per FID ---
    print("[INFO] Pre-calcolo statistiche Inception reali per FID...")
    real_mu, real_sigma = compute_real_stats(train_loader, device,
                                             num_samples=FID_NUM_SAMPLES)

    # --- Training loop ---
    best_val_loss = float('inf')
    best_fid = float('inf')
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, noise_schedule, optimizer, scaler, device)

        # EMA update
        ema.update(model)

        # Val loss con EMA
        ema.apply_shadow(model)
        val_loss = evaluate(model, test_loader, noise_schedule, device)

        # FID / IS ogni FID_EVERY epoche + ultima epoca
        fid_val = None
        is_mean = None
        is_std = None
        do_fid = ((epoch + 1) % FID_EVERY == 0) or (epoch + 1 == epochs)

        if do_fid:
            print(f"[FID] Generazione {FID_NUM_SAMPLES} campioni per FID/IS...")
            gen_start = time.time()
            samples = generate_samples_batched(
                model, noise_schedule, FID_NUM_SAMPLES, device, batch_size=256)
            gen_time = time.time() - gen_start
            print(f"[FID] Campioni generati in {gen_time:.1f}s")

            print("[FID] Calcolo features Inception...")
            gen_features, gen_probs = get_inception_features(samples, device)
            gen_features_np = gen_features.numpy()
            gen_probs_np = gen_probs.numpy()

            gen_mu = np.mean(gen_features_np, axis=0)
            gen_sigma = np.cov(gen_features_np, rowvar=False)
            fid_val = compute_fid(real_mu, real_sigma, gen_mu, gen_sigma)
            is_mean, is_std = compute_inception_score(gen_probs_np)

            if fid_val < best_fid:
                best_fid = fid_val

            log_data["fid_log"].append({
                "epoch": epoch + 1,
                "fid": round(fid_val, 2),
                "is_mean": round(is_mean, 2),
                "is_std": round(is_std, 2),
                "gen_time_sec": round(gen_time, 1),
            })
            print(f"[FID] Epoch {epoch+1}: FID={fid_val:.2f}, "
                  f"IS={is_mean:.2f}±{is_std:.2f}")

            del samples, gen_features, gen_probs
            torch.cuda.empty_cache()

        # Campioni visivi ogni SAMPLE_EVERY epoche + ultima
        do_sample = ((epoch + 1) % SAMPLE_EVERY == 0) or (epoch + 1 == epochs)
        if do_sample:
            n_vis = SAMPLE_GRID * SAMPLE_GRID
            vis_samples = generate_samples_batched(
                model, noise_schedule, n_vis, device, batch_size=n_vis)
            # [-1, 1] → [0, 1]
            vis_samples = (vis_samples + 1.0) / 2.0
            vis_samples = vis_samples.clamp(0, 1)

            grid = torchvision.utils.make_grid(
                vis_samples, nrow=SAMPLE_GRID, padding=2, normalize=False)
            sample_path = os.path.join(
                results_dir,
                f"dit_cifar10_samples_{scale}_{activation}_epoch{epoch+1}.png")
            torchvision.utils.save_image(grid, sample_path)
            print(f"[SAMPLE] Salvati campioni: {sample_path}")

            del vis_samples
            torch.cuda.empty_cache()

        # Ripristina pesi originali (dopo aver usato EMA per val/gen)
        ema.restore(model)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Epoch log
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": round(current_lr, 8),
            "epoch_time_sec": round(epoch_time, 1),
        }
        if fid_val is not None:
            epoch_log["fid"] = round(fid_val, 2)
            epoch_log["is_mean"] = round(is_mean, 2)
            epoch_log["is_std"] = round(is_std, 2)

        # β tracking per NOVA
        if activation == "nova":
            for block in model.blocks:
                for m in block.mlp.modules():
                    if isinstance(m, (NOVACuda, NOVAPython)):
                        epoch_log["beta"] = round(m.beta.item(), 6)
                        break
                if "beta" in epoch_log:
                    break

        log_data["epoche_log"].append(epoch_log)

        # Salvataggio incrementale
        log_data["metriche"] = {
            "best_val_loss": round(best_val_loss, 6),
            "final_val_loss": round(val_loss, 6),
            "final_train_loss": round(train_loss, 6),
            "best_fid": round(best_fid, 2) if best_fid < float('inf') else None,
        }
        log_data["tempo_totale_sec"] = round(time.time() - start_time, 2)
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        fid_str = f"  FID={fid_val:.2f}" if fid_val is not None else ""
        beta_str = (f"  β={epoch_log.get('beta', '')}"
                    if activation == "nova" else "")
        print(f"[{scale.upper()}/{activation.upper()}] "
              f"Epoch {epoch+1:03d}/{epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={current_lr:.6f}  ({epoch_time:.1f}s)"
              f"{fid_str}{beta_str}")

    elapsed = time.time() - start_time

    # --- Salvataggio finale ---
    log_data["tempo_totale_sec"] = round(elapsed, 2)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    print(f"\n{'='*60}")
    print(f"  RISULTATI — {scale.upper()} / {activation.upper()}")
    print(f"  Parametri: {num_params:,}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    if best_fid < float('inf'):
        print(f"  Best FID: {best_fid:.2f}")
    print(f"  Tempo totale: {elapsed:.1f}s")
    print(f"  Log: {log_path}")
    print(f"{'='*60}\n")

    # --- Cleanup VRAM ---
    del model, optimizer, scaler, scheduler, ema
    del train_loader, test_loader
    torch.cuda.empty_cache()
    print(f"[{scale.upper()}/{activation.upper()}] VRAM liberata.")


# ==============================================================
# PLOT PER IL PAPER (7 plot)
# ==============================================================

def generate_plots():
    """Genera tutti i plot per il paper dai log JSON in results/."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results_dir = _get_results_dir()
    scales = ["tiny", "small", "base"]
    activations = ALL_ACTIVATIONS

    # --- Carica log ---
    def load_latest(pattern):
        files = sorted(glob.glob(os.path.join(results_dir, pattern)))
        if not files:
            return None
        with open(files[-1]) as f:
            return json.load(f)

    data = {}
    for scale in scales:
        for act in activations:
            d = load_latest(f"dit_cifar10_{scale}_{act}_*.json")
            if d:
                data[(scale, act)] = d

    if not data:
        print("[PLOT] Nessun risultato DiT trovato. Esegui prima gli esperimenti.")
        return

    colors = {"nova": "#E63946", "gelu": "#457B9D", "silu": "#2A9D8F",
              "mish": "#E9C46A", "relu": "#264653"}
    scale_labels = {
        "tiny": "Tiny (~1.5M)",
        "small": "Small (~4M)",
        "base": "Base (~10M)",
    }

    # ==========================================================
    # PLOT 1: Training curves per scala (2×3): train loss + val loss
    # ==========================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("DiT DDPM CIFAR-10 — Training Curves",
                 fontsize=14, fontweight='bold')

    for col, scale in enumerate(scales):
        ax_train = axes[0, col]
        ax_val = axes[1, col]

        for act in activations:
            key = (scale, act)
            if key not in data:
                continue
            d = data[key]
            ep = [e["epoch"] for e in d["epoche_log"]]
            tl = [e["train_loss"] for e in d["epoche_log"]]
            vl = [e["val_loss"] for e in d["epoche_log"]]

            label = act.upper()
            ax_train.plot(ep, tl, color=colors.get(act, "gray"),
                          linewidth=1.5, label=label)
            best_vl = d["metriche"].get("best_val_loss", min(vl))
            ax_val.plot(ep, vl, color=colors.get(act, "gray"),
                        linewidth=1.5,
                        label=f"{label} (best {best_vl:.4f})")

        ax_train.set_title(f"{scale_labels.get(scale, scale)} — Train Loss",
                           fontsize=11)
        ax_train.set_ylabel("MSE Loss" if col == 0 else "")
        ax_train.legend(fontsize=8)
        ax_train.grid(True, alpha=0.3)

        ax_val.set_title(f"{scale_labels.get(scale, scale)} — Val Loss",
                         fontsize=11)
        ax_val.set_xlabel("Epoca")
        ax_val.set_ylabel("MSE Loss" if col == 0 else "")
        ax_val.legend(fontsize=8)
        ax_val.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "plot_dit_cifar10_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # ==========================================================
    # PLOT 2: Scaling curve — best val loss vs parametri
    # ==========================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("DiT Scaling: Best Val Loss vs Model Size",
                 fontsize=13, fontweight='bold')

    for act in activations:
        params_list = []
        loss_list = []
        for scale in scales:
            key = (scale, act)
            if key not in data:
                continue
            d = data[key]
            params_list.append(d["configurazione"]["num_params"] / 1e6)
            loss_list.append(d["metriche"]["best_val_loss"])
        if params_list:
            ax.plot(params_list, loss_list, color=colors.get(act, "gray"),
                    marker='o', markersize=8, linewidth=2,
                    label=act.upper())

    ax.set_xlabel("Parametri (M)", fontsize=11)
    ax.set_ylabel("Best Val MSE Loss", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "plot_dit_cifar10_scaling_loss.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # ==========================================================
    # PLOT 3: FID curves per scala (1×3)
    # ==========================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("DiT DDPM CIFAR-10 — FID nel Tempo",
                 fontsize=14, fontweight='bold')

    for col, scale in enumerate(scales):
        ax = axes[col]
        for act in activations:
            key = (scale, act)
            if key not in data:
                continue
            d = data[key]
            fid_log = d.get("fid_log", [])
            if not fid_log:
                continue
            ep = [f["epoch"] for f in fid_log]
            fids = [f["fid"] for f in fid_log]
            ax.plot(ep, fids, color=colors.get(act, "gray"),
                    marker='o', markersize=5, linewidth=1.5,
                    label=f"{act.upper()} (best {min(fids):.1f})")

        ax.set_title(scale_labels.get(scale, scale), fontsize=11)
        ax.set_xlabel("Epoca")
        ax.set_ylabel("FID" if col == 0 else "")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "plot_dit_cifar10_fid_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # ==========================================================
    # PLOT 4: FID scaling — best FID vs parametri
    # ==========================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("DiT Scaling: Best FID vs Model Size",
                 fontsize=13, fontweight='bold')

    for act in activations:
        params_list = []
        fid_list = []
        for scale in scales:
            key = (scale, act)
            if key not in data:
                continue
            d = data[key]
            fid_log = d.get("fid_log", [])
            if not fid_log:
                continue
            best_fid = min(f["fid"] for f in fid_log)
            params_list.append(d["configurazione"]["num_params"] / 1e6)
            fid_list.append(best_fid)
        if params_list:
            ax.plot(params_list, fid_list, color=colors.get(act, "gray"),
                    marker='s', markersize=8, linewidth=2,
                    label=act.upper())

    ax.set_xlabel("Parametri (M)", fontsize=11)
    ax.set_ylabel("Best FID (↓ is better)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "plot_dit_cifar10_fid_scaling.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # ==========================================================
    # PLOT 5: IS scaling — best IS vs parametri
    # ==========================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("DiT Scaling: Best IS vs Model Size",
                 fontsize=13, fontweight='bold')

    for act in activations:
        params_list = []
        is_list = []
        for scale in scales:
            key = (scale, act)
            if key not in data:
                continue
            d = data[key]
            fid_log = d.get("fid_log", [])
            if not fid_log:
                continue
            # Best IS = max IS mean
            best_is = max(f["is_mean"] for f in fid_log)
            params_list.append(d["configurazione"]["num_params"] / 1e6)
            is_list.append(best_is)
        if params_list:
            ax.plot(params_list, is_list, color=colors.get(act, "gray"),
                    marker='D', markersize=8, linewidth=2,
                    label=act.upper())

    ax.set_xlabel("Parametri (M)", fontsize=11)
    ax.set_ylabel("Best IS (↑ is better)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "plot_dit_cifar10_is_scaling.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # ==========================================================
    # PLOT 6: β evolution per scala (NOVA only)
    # ==========================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("DiT — Evoluzione del parametro β (NOVA)",
                 fontsize=13, fontweight='bold')

    scale_colors = {"tiny": "#264653", "small": "#2A9D8F", "base": "#E76F51"}
    for scale in scales:
        key = (scale, "nova")
        if key not in data:
            continue
        d = data[key]
        ep = [e["epoch"] for e in d["epoche_log"]]
        betas = [e.get("beta", None) for e in d["epoche_log"]]
        if betas[0] is not None:
            ax.plot(ep, betas, color=scale_colors.get(scale, "gray"),
                    linewidth=2, label=scale_labels.get(scale, scale))

    ax.set_xlabel("Epoca", fontsize=11)
    ax.set_ylabel("β", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "plot_dit_cifar10_beta_evolution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # ==========================================================
    # PLOT 7: Sample grid comparison (ultima epoca disponibile)
    # ==========================================================
    try:
        from PIL import Image

        fig, axes_grid = plt.subplots(1, len(activations),
                                      figsize=(6 * len(activations), 6))
        if len(activations) == 1:
            axes_grid = [axes_grid]
        fig.suptitle("DiT DDPM CIFAR-10 — Campioni Generati (ultima epoca)",
                     fontsize=14, fontweight='bold')

        # Usa la scala "small" come default per la comparazione
        comp_scale = "small" if ("small", activations[0]) in data else scales[0]

        for idx, act in enumerate(activations):
            ax = axes_grid[idx]
            # Trova il file campione più recente
            sample_files = sorted(glob.glob(os.path.join(
                results_dir,
                f"dit_cifar10_samples_{comp_scale}_{act}_epoch*.png")))
            if sample_files:
                img = Image.open(sample_files[-1])
                ax.imshow(np.array(img))
                ax.set_title(f"{act.upper()} — {comp_scale.capitalize()}",
                             fontsize=12)
            else:
                ax.text(0.5, 0.5, "No samples", ha='center', va='center',
                        transform=ax.transAxes, fontsize=14)
                ax.set_title(f"{act.upper()} — no data", fontsize=12)
            ax.axis('off')

        plt.tight_layout()
        path = os.path.join(results_dir,
                            "plot_dit_cifar10_sample_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[PLOT] Salvato: {path}")
    except ImportError:
        print("[PLOT] PIL non disponibile, skip sample comparison plot.")

    print(f"\n[PLOT] Tutti i plot DiT generati in: {results_dir}")


# ==============================================================
# LAUNCHER: parallelo su 2 GPU, per scala
# ==============================================================

def launch_all(scales=None, activations=None):
    global _nova_cuda_ext

    if scales is None:
        scales = list(SCALING_CONFIGS.keys())
    if activations is None:
        activations = ALL_ACTIVATIONS

    # 1. Pre-download dataset
    print("[LAUNCHER] Pre-download dataset CIFAR-10...")
    data_root = _get_data_root()
    torchvision.datasets.CIFAR10(root=data_root, train=True, download=True)
    torchvision.datasets.CIFAR10(root=data_root, train=False, download=True)
    print("[LAUNCHER] Dataset pronto.")

    # 2. Pre-compilazione kernel CUDA
    if "nova" in activations:
        print("[LAUNCHER] Pre-compilazione kernel CUDA NOVA (30-60s)...")
        try:
            _nova_cuda_ext = _compile_nova_cuda()
            print("[LAUNCHER] Kernel CUDA compilato e cachato.")
        except Exception as e:
            print(f"[LAUNCHER] ATTENZIONE: Compilazione CUDA fallita: {e}")
            print("[LAUNCHER] NOVA userà il fallback Python puro.")

    # 3. Per ogni scala, lancia le attivazioni a coppie sulle 2 GPU
    script = os.path.abspath(__file__)

    for scale in scales:
        cfg = SCALING_CONFIGS[scale]
        print(f"\n{'#'*60}")
        print(f"  SCALA: {scale.upper()} ({cfg['num_layers']}L, "
              f"{cfg['embed_dim']}d, {cfg['num_heads']}h)")
        print(f"  Diffusion: T={T_DIFFUSION}, Cosine schedule")
        print(f"{'#'*60}")

        pairs = [(activations[i],
                  activations[i + 1] if i + 1 < len(activations) else None)
                 for i in range(0, len(activations), 2)]

        for pair in pairs:
            procs = []
            for gpu_id, act in enumerate(pair):
                if act is None:
                    continue
                cmd = [sys.executable, script,
                       "--scale", scale,
                       "--activation", act,
                       "--gpu", str(gpu_id)]
                print(f"\n[LAUNCHER] Avvio: {' '.join(cmd)}")
                p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
                procs.append((act, p))

            for act, p in procs:
                p.wait()
                if p.returncode != 0:
                    print(f"[LAUNCHER] ERRORE: {scale}/{act} "
                          f"terminato con codice {p.returncode}")
                else:
                    print(f"[LAUNCHER] {scale.upper()}/{act.upper()} "
                          f"completato con successo.")

    # 4. Genera i plot
    print("\n[LAUNCHER] Generazione plot per il paper...")
    generate_plots()

    print(f"\n{'='*60}")
    print("  DiT DDPM CIFAR-10 — STUDIO COMPLETATO")
    print(f"{'='*60}")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DiT DDPM su CIFAR-10: NOVA vs GELU")
    parser.add_argument("--scale", type=str,
                        choices=list(SCALING_CONFIGS.keys()),
                        help="Scala del modello (ometti per tutte)")
    parser.add_argument("--activation", type=str,
                        choices=["nova", "gelu", "silu", "mish", "relu"],
                        help="Funzione di attivazione (ometti per NOVA+GELU)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="ID della GPU (default: 0)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Solo generazione plot (senza training)")
    args = parser.parse_args()

    if args.plot_only:
        generate_plots()
    elif args.scale is not None and args.activation is not None:
        run_experiment(args.scale, args.activation, args.gpu)
    elif args.activation is not None and args.scale is None:
        for scale in SCALING_CONFIGS:
            run_experiment(scale, args.activation, args.gpu)
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            print(f"[ATTENZIONE] Solo {n_gpus} GPU disponibili. "
                  "Lancia manualmente con --scale, --activation e --gpu.")
            sys.exit(1)
        scales = [args.scale] if args.scale else None
        launch_all(scales=scales)
