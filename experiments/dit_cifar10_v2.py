#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESPERIMENTO v2: DiT DDPM su CIFAR-10 — Solo Base
===================================================
Diffusion Transformer (DiT, Peebles & Xie 2023) su CIFAR-10 con DDPM.

Miglioramenti rispetto a v1:
    1. 400 epoche (da 100) — il modello era gravemente sotto-addestrato
    2. EMA decay 0.9995 (da 0.9999) — più reattivo con training medio
    3. DDIM sampler (250 step, η=0) per FID — 4× più veloce e FID migliore
    4. RandomCrop(32, padding=4) — augmentation standard CIFAR-10
    5. weight_decay=0.01 (da 0.0) — regolarizzazione Transformer
    6. FID ogni 25 epoche (da 50) — curve più informative per il paper
    7. Warmup 10 epoche (da 5) — più stabile con training lungo

Architettura (Base):
    8L, 384d, 6h, MLP×4, ~22M params, batch 128
    Cosine schedule, T=1000, MSE ε-prediction, AdamW lr=3e-4

Uso:
    # Singolo esperimento:
    python dit_cifar10_v2.py --activation nova --gpu 0
    python dit_cifar10_v2.py --activation gelu --gpu 0

    # Solo generazione plot:
    python dit_cifar10_v2.py --plot-only
"""

import argparse
import subprocess
import sys
import os
import math
import time
import json
import glob
import random
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
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 64
DIFFUSION_TIMESTEPS = 1000

# --- v2: parametri migliorati ---
EMA_DECAY = 0.9995          # v1: 0.9999 — troppo lento per <100K step
MAX_GRAD_NORM = 1.0
FID_NUM_SAMPLES = 10000
FID_BATCH_SIZE = 256
FID_EVERY_N_EPOCHS = 25     # v1: 50 — curve più dense
SAMPLE_GRID_SIZE = 8        # 8×8 = 64 campioni visivi
DDIM_STEPS = 250            # NUOVO: DDIM per sampling veloce e migliore


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


# --- Solo Base ---
BASE_CONFIG = {
    "embed_dim": 384,
    "num_heads": 6,
    "num_layers": 8,
    "batch_size": 128,
    "lr": 3e-4,
    "weight_decay": 0.01,      # v1: 0.0
    "epochs": 400,             # v1: 100
    "warmup_epochs": 10,       # v1: 5
}

ALL_ACTIVATIONS = ["nova", "gelu"]


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
        name='nova_cuda_ext_dit_v2',
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
    """Factory per attivazioni. Restituisce (module, backend_info)."""
    if name == "nova":
        return make_nova(beta=beta)
    mapping = {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "relu": nn.ReLU,
    }
    if name not in mapping:
        raise ValueError(f"Attivazione non supportata: {name}")
    return mapping[name](), name


# ==============================================================
# DIFFUSION: COSINE NOISE SCHEDULE
# ==============================================================

class CosineNoiseSchedule:
    """Cosine schedule da Nichol & Dhariwal (2021), Improved DDPM."""

    def __init__(self, timesteps=1000, s=0.008, device=None):
        self.T = timesteps
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0, 0.999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        if device is not None:
            self.to(device)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        return self

    def q_sample(self, x_0, t, noise=None):
        """Forward process: q(x_t | x_0) = N(sqrt(ᾱ_t) x_0, (1-ᾱ_t) I)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise


# ==============================================================
# EMA (Exponential Moving Average)
# ==============================================================

class EMA:
    """Exponential Moving Average dei pesi del modello."""

    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay)

    def apply(self, model):
        """Applica i pesi EMA al modello, salvando gli originali."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Ripristina i pesi originali del modello."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ==============================================================
# DiT ARCHITECTURE
# ==============================================================

class SinusoidalTimestepEmbedding(nn.Module):
    """Embedding sinusoidale per il timestep, stile Transformer/DDPM."""

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
    """sinusoidal → Linear → SiLU → Linear (SiLU fisso, standard DiT)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.sinusoidal = SinusoidalTimestepEmbedding(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        return self.mlp(self.sinusoidal(t))


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


class DiTBlock(nn.Module):
    """Blocco DiT con AdaLN-Zero (Peebles & Xie 2023)."""

    def __init__(self, dim, num_heads, mlp_ratio, act_layer):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            act_layer,
            nn.Linear(mlp_hidden, dim),
        )
        # AdaLN modulation: SiLU → Linear → 6 * dim
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        # Inizializzazione zero per i gate (DiT-Zero)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        mod = self.adaLN_modulation(c).unsqueeze(1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        h = self.norm1(x)
        h = h * (1 + gamma1) + beta1
        h = self.attn(h)
        x = x + alpha1 * h

        h = self.norm2(x)
        h = h * (1 + gamma2) + beta2
        h = self.mlp(h)
        x = x + alpha2 * h

        return x


class FinalLayer(nn.Module):
    """Layer finale del DiT: AdaLN → Linear → unpatchify."""

    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        mod = self.adaLN_modulation(c).unsqueeze(1)
        gamma, beta = mod.chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + gamma) + beta
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """Diffusion Transformer (DiT) per CIFAR-10 32×32."""

    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=192, num_heads=3, num_layers=4,
                 mlp_ratio=4, act_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim))

        self.time_embed = TimestepEmbedder(embed_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio, act_layer)
            for _ in range(num_layers)
        ])

        self.final_layer = FinalLayer(embed_dim, patch_size, in_channels)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def unpatchify(self, x):
        B = x.shape[0]
        p = self.patch_size
        c = self.in_channels
        h = w = int(self.num_patches ** 0.5)
        x = x.reshape(B, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, c, h * p, w * p)
        return x

    def forward(self, x, t):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        c = self.time_embed(t)

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x


# ==============================================================
# FID / IS COMPUTATION (Inception v3)
# ==============================================================

class InceptionFeatureExtractor(nn.Module):
    """Estrae features (2048-dim) e logits (1000-dim) da Inception v3."""

    def __init__(self, device):
        super().__init__()
        inception = torchvision.models.inception_v3(
            weights='IMAGENET1K_V1')
        self.fc_weight = inception.fc.weight.data.clone().to(device)
        self.fc_bias = inception.fc.bias.data.clone().to(device)
        inception.fc = nn.Identity()
        inception.aux_logits = False
        self.model = inception.to(device).eval()
        self.device = device

        self.register_buffer('mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    @torch.no_grad()
    def forward(self, images):
        x = F.interpolate(images, size=(299, 299),
                          mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        features = self.model(x)
        logits = F.linear(features, self.fc_weight, self.fc_bias)
        return features, logits


def compute_inception_features(images, extractor, batch_size=64):
    all_features = []
    all_probs = []
    N = images.shape[0]

    for i in range(0, N, batch_size):
        batch = images[i:i+batch_size].to(extractor.device)
        features, logits = extractor(batch)
        all_features.append(features.cpu())
        all_probs.append(F.softmax(logits, dim=1).cpu())

    return torch.cat(all_features, dim=0), torch.cat(all_probs, dim=0)


def compute_fid(mu1, sigma1, mu2, sigma2):
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return fid


def compute_inception_score(probs, splits=10):
    N = probs.shape[0]
    split_size = N // splits
    scores = []
    for k in range(splits):
        part = probs[k * split_size: (k + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))
    return float(np.mean(scores)), float(np.std(scores))


def compute_real_stats(data_loader, extractor, device, max_samples=50000):
    all_features = []
    count = 0

    for images, _ in data_loader:
        images = (images + 1.0) / 2.0
        images = images.to(device)
        features, _ = extractor(images)
        all_features.append(features.cpu().numpy())
        count += images.shape[0]
        if count >= max_samples:
            break

    features = np.concatenate(all_features, axis=0)[:max_samples]
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


# ==============================================================
# DDIM SAMPLING (NUOVO in v2)
# ==============================================================

@torch.no_grad()
def ddim_sample(model, noise_schedule, num_samples, device,
                ddim_steps=250, eta=0.0,
                img_size=32, channels=3, batch_size=256):
    """Genera campioni con DDIM (Song et al., 2020).

    Vantaggi rispetto a DDPM:
        - ~4× più veloce (250 step vs 1000)
        - FID tipicamente uguale o migliore
        - eta=0 → deterministico (più stabile per FID)
    """
    model.eval()
    all_samples = []

    # Subsequence di timestep uniformemente spaziati
    step_indices = torch.linspace(
        0, noise_schedule.T - 1, ddim_steps, dtype=torch.long)

    for start in range(0, num_samples, batch_size):
        B = min(batch_size, num_samples - start)
        x = torch.randn(B, channels, img_size, img_size, device=device)

        for i in reversed(range(len(step_indices))):
            t_idx = int(step_indices[i].item())
            t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)

            with autocast(device_type='cuda', dtype=torch.float16):
                eps_pred = model(x, t_batch)
            eps_pred = eps_pred.float()

            alpha_bar_t = noise_schedule.alphas_cumprod[t_idx]

            # Predicted x_0
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            if i > 0:
                t_prev = int(step_indices[i - 1].item())
                alpha_bar_prev = noise_schedule.alphas_cumprod[t_prev]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            # DDIM update
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                * (1 - alpha_bar_t / alpha_bar_prev))
            dir_xt = torch.sqrt(
                torch.clamp(1 - alpha_bar_prev - sigma**2, min=0.0)) * eps_pred
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt

            if i > 0 and eta > 0:
                x = x + sigma * torch.randn_like(x)

        x = torch.clamp(x, -1.0, 1.0)
        all_samples.append(x.cpu())

    return torch.cat(all_samples, dim=0)[:num_samples]


# Manteniamo anche DDPM per confronto (opzionale)
@torch.no_grad()
def ddpm_sample(model, noise_schedule, num_samples, device,
                img_size=32, channels=3, batch_size=256):
    """Genera campioni con DDPM reverse process (legacy)."""
    model.eval()
    all_samples = []

    for start in range(0, num_samples, batch_size):
        B = min(batch_size, num_samples - start)
        x = torch.randn(B, channels, img_size, img_size, device=device)

        for t_idx in reversed(range(noise_schedule.T)):
            t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)

            with autocast(device_type='cuda', dtype=torch.float16):
                eps_pred = model(x, t_batch)
            eps_pred = eps_pred.float()

            alpha_t = noise_schedule.alphas[t_idx]
            alpha_bar_t = noise_schedule.alphas_cumprod[t_idx]
            beta_t = noise_schedule.betas[t_idx]

            mu = noise_schedule.sqrt_recip_alphas[t_idx] * (
                x - beta_t / noise_schedule.sqrt_one_minus_alphas_cumprod[t_idx] * eps_pred
            )

            if t_idx > 0:
                sigma = torch.sqrt(beta_t)
                z = torch.randn_like(x)
                x = mu + sigma * z
            else:
                x = mu

        x = torch.clamp(x, -1.0, 1.0)
        all_samples.append(x.cpu())

    return torch.cat(all_samples, dim=0)[:num_samples]


def save_sample_grid(samples, path, nrow=8):
    from torchvision.utils import save_image
    save_image((samples + 1.0) / 2.0, path, nrow=nrow, padding=2)


# ==============================================================
# DATI: CIFAR-10 (normalizzato a [-1, 1])
# ==============================================================

def get_cifar10_loaders(batch_size, num_workers):
    """CIFAR-10 normalizzato a [-1, 1] per diffusion.

    v2: aggiunto RandomCrop(32, padding=4) — augmentation standard CIFAR-10.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),   # NUOVO v2
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data_root = _get_data_root()
    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


# ==============================================================
# TRAINING & VALUTAZIONE
# ==============================================================

def train_one_epoch(model, loader, optimizer, scaler, noise_schedule, device, ema):
    model.train()
    running_loss = 0.0
    total = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        B = images.shape[0]

        t = torch.randint(0, noise_schedule.T, (B,), device=device, dtype=torch.long)

        noise = torch.randn_like(images)
        x_noisy = noise_schedule.q_sample(images, t, noise=noise)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=torch.float16):
            eps_pred = model(x_noisy, t)
            loss = F.mse_loss(eps_pred, noise)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        running_loss += loss.item() * B
        total += B

    return running_loss / total


@torch.no_grad()
def evaluate(model, loader, noise_schedule, device):
    model.eval()
    running_loss = 0.0
    total = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        B = images.shape[0]

        t = torch.randint(0, noise_schedule.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(images)
        x_noisy = noise_schedule.q_sample(images, t, noise=noise)

        with autocast(device_type='cuda', dtype=torch.float16):
            eps_pred = model(x_noisy, t)
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

def run_experiment(activation: str, gpu: int) -> None:
    set_seed(SEED)

    cfg = BASE_CONFIG
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    embed_dim = cfg["embed_dim"]
    num_heads = cfg["num_heads"]
    num_layers = cfg["num_layers"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    epochs = cfg["epochs"]
    warmup_epochs = cfg["warmup_epochs"]

    scale = "base"
    experiment_name = f"dit_cifar10_{scale}_{activation}"
    model_desc = (f"DiT-{scale.capitalize()} ({num_layers}L, {embed_dim}d, "
                  f"{num_heads}h, MLP×{MLP_RATIO}) + {activation.upper()}")

    print(f"\n{'='*60}")
    print(f"  ESPERIMENTO v2: {experiment_name}")
    print(f"  GPU: {gpu} ({torch.cuda.get_device_name(device)})")
    print(f"  Attivazione: {activation.upper()}")
    print(f"  Architettura: {num_layers}L, {embed_dim}d, {num_heads}h")
    print(f"  Epoche: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  Diffusion: T={DIFFUSION_TIMESTEPS}, Cosine schedule, MSE ε-pred")
    print(f"  Sampler: DDIM ({DDIM_STEPS} step, η=0)")
    print(f"  EMA decay: {EMA_DECAY}")
    print(f"  Augmentation: RandomHorizontalFlip + RandomCrop(32, pad=4)")
    print(f"  Mixed Precision: FP16 | Seed: {SEED}")
    print(f"{'='*60}\n")

    # --- Activation layer ---
    nova_backend = None
    act_layer, backend_info = make_activation(activation, beta=1.0)
    if activation == "nova":
        nova_backend = backend_info
        print(f"[INFO] NOVA backend: {nova_backend}")

    # --- Modello DiT ---
    model = DiT(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=IMG_CHANNELS,
        embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
        mlp_ratio=MLP_RATIO, act_layer=act_layer,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parametri totali: {num_params:,}")

    # --- Noise schedule ---
    noise_schedule = CosineNoiseSchedule(
        timesteps=DIFFUSION_TIMESTEPS, device=device)

    # --- EMA ---
    ema = EMA(model, decay=EMA_DECAY)

    # --- Dati CIFAR-10 ---
    train_loader, test_loader = get_cifar10_loaders(batch_size, NUM_WORKERS)

    # --- Ottimizzatore, Scheduler, Scaler ---
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)     # v2: wd=0.01
    scheduler = build_scheduler(optimizer, warmup_epochs, epochs)
    scaler = GradScaler()

    # --- Log setup ---
    results_dir = _get_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(results_dir, f"{experiment_name}_{timestamp}.json")

    log_data = {
        "esperimento": experiment_name,
        "versione": "v2",
        "data": datetime.now().strftime("%Y-%m-%d"),
        "hardware": f"{HARDWARE} (GPU {gpu})",
        "seed": SEED,
        "dataset": DATASET,
        "modello": model_desc,
        "obiettivo": (f"DiT DDPM v2 su CIFAR-10: base DiT con {activation}, "
                      f"T={DIFFUSION_TIMESTEPS}, {epochs} epoche, "
                      f"DDIM {DDIM_STEPS} step"),
        "miglioramenti_v2": [
            f"epochs: 100 → {epochs}",
            f"ema_decay: 0.9999 → {EMA_DECAY}",
            f"weight_decay: 0.0 → {weight_decay}",
            f"warmup_epochs: 5 → {warmup_epochs}",
            f"sampler: DDPM 1000 step → DDIM {DDIM_STEPS} step",
            "augmentation: +RandomCrop(32, padding=4)",
            f"fid_every: 50 → {FID_EVERY_N_EPOCHS} epoche",
        ],
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
            "mixed_precision": "fp16",
            "num_params": num_params,
            "diffusion_timesteps": DIFFUSION_TIMESTEPS,
            "noise_schedule": "cosine",
            "ema_decay": EMA_DECAY,
            "grad_clip": MAX_GRAD_NORM,
            "fid_num_samples": FID_NUM_SAMPLES,
            "sampler": f"DDIM {DDIM_STEPS} step, eta=0",
            "augmentation": "RandomHorizontalFlip + RandomCrop(32, padding=4)",
        },
        "metriche": {},
        "epoche_log": [],
        "fid_is_log": [],
        "tempo_totale_sec": None,
    }
    if nova_backend:
        log_data["configurazione"]["nova_backend"] = nova_backend

    # --- Pre-compute real stats per FID (una volta sola) ---
    print("[INFO] Calcolo statistiche Inception per dati reali (una tantum)...")
    inception_extractor = InceptionFeatureExtractor(device)
    real_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=FID_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)
    real_mu, real_sigma = compute_real_stats(
        real_loader, inception_extractor, device)
    print("[INFO] Statistiche reali calcolate.")
    del real_loader

    # --- Training loop ---
    best_val_loss = float('inf')
    best_fid = float('inf')
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, noise_schedule, device, ema)

        # Validazione con EMA
        ema.apply(model)
        val_loss = evaluate(model, test_loader, noise_schedule, device)
        ema.restore(model)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # --- Epoch log ---
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": round(current_lr, 8),
            "epoch_time_sec": round(epoch_time, 1),
        }
        if activation == "nova":
            for m in model.modules():
                if isinstance(m, (NOVACuda, NOVAPython)):
                    epoch_log["beta"] = round(m.beta.item(), 6)
                    break

        log_data["epoche_log"].append(epoch_log)

        print_str = (
            f"[BASE/{activation.upper()}] "
            f"Epoch {epoch+1:03d}/{epochs}  "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"lr={current_lr:.6f}  ({epoch_time:.1f}s)"
        )
        if activation == "nova" and "beta" in epoch_log:
            print_str += f"  β={epoch_log['beta']}"
        print(print_str)

        # --- FID/IS ogni N epoche e all'ultima ---
        is_fid_epoch = (
            (epoch + 1) % FID_EVERY_N_EPOCHS == 0) or (epoch + 1 == epochs)
        if is_fid_epoch:
            print(f"[FID/IS] Generazione {FID_NUM_SAMPLES} campioni con "
                  f"DDIM ({DDIM_STEPS} step)...")
            fid_start = time.time()

            # Genera campioni con EMA + DDIM
            ema.apply(model)
            samples = ddim_sample(
                model, noise_schedule, FID_NUM_SAMPLES, device,
                ddim_steps=DDIM_STEPS, eta=0.0,
                img_size=IMG_SIZE, channels=IMG_CHANNELS,
                batch_size=FID_BATCH_SIZE)
            ema.restore(model)

            # Salva griglia campioni
            grid_samples = samples[:SAMPLE_GRID_SIZE ** 2]
            grid_path = os.path.join(
                results_dir,
                f"dit_cifar10_samples_base_{activation}_epoch{epoch+1:03d}.png")
            save_sample_grid(grid_samples, grid_path, nrow=SAMPLE_GRID_SIZE)
            print(f"[FID/IS] Campioni salvati: {grid_path}")

            # Calcola features Inception
            samples_01 = (samples + 1.0) / 2.0
            gen_features, gen_probs = compute_inception_features(
                samples_01, inception_extractor, batch_size=64)

            # FID
            gen_features_np = gen_features.numpy()
            gen_mu = np.mean(gen_features_np, axis=0)
            gen_sigma = np.cov(gen_features_np, rowvar=False)
            fid_value = compute_fid(real_mu, real_sigma, gen_mu, gen_sigma)

            # IS
            gen_probs_np = gen_probs.numpy()
            is_mean, is_std = compute_inception_score(gen_probs_np)

            if fid_value < best_fid:
                best_fid = fid_value

            fid_time = time.time() - fid_start
            fid_log = {
                "epoch": epoch + 1,
                "fid": round(fid_value, 2),
                "is_mean": round(is_mean, 2),
                "is_std": round(is_std, 2),
                "sampler": f"DDIM-{DDIM_STEPS}",
                "generation_time_sec": round(fid_time, 1),
            }
            log_data["fid_is_log"].append(fid_log)

            print(f"[FID/IS] Epoch {epoch+1}: FID={fid_value:.2f}, "
                  f"IS={is_mean:.2f}±{is_std:.2f} ({fid_time:.1f}s)")

            del samples, samples_01, gen_features, gen_probs
            torch.cuda.empty_cache()

        # --- Salvataggio incrementale ---
        log_data["metriche"] = {
            "best_val_loss": round(best_val_loss, 6),
            "final_val_loss": round(val_loss, 6),
            "final_train_loss": round(train_loss, 6),
        }
        if log_data["fid_is_log"]:
            log_data["metriche"]["best_fid"] = round(best_fid, 2)
            log_data["metriche"]["latest_fid"] = log_data["fid_is_log"][-1]["fid"]
            log_data["metriche"]["latest_is_mean"] = (
                log_data["fid_is_log"][-1]["is_mean"])
            log_data["metriche"]["latest_is_std"] = (
                log_data["fid_is_log"][-1]["is_std"])
        log_data["tempo_totale_sec"] = round(time.time() - start_time, 2)

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

    elapsed = time.time() - start_time

    # --- Salvataggio finale ---
    log_data["tempo_totale_sec"] = round(elapsed, 2)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    print(f"\n{'='*60}")
    print(f"  RISULTATI v2 — BASE / {activation.upper()}")
    print(f"  Parametri: {num_params:,}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    if log_data["fid_is_log"]:
        print(f"  Best FID: {best_fid:.2f}")
        final_fid = log_data["fid_is_log"][-1]
        print(f"  Final FID: {final_fid['fid']:.2f}, "
              f"IS: {final_fid['is_mean']:.2f}±{final_fid['is_std']:.2f}")
    print(f"  Tempo totale: {elapsed:.1f}s ({elapsed/3600:.1f}h)")
    print(f"  Log: {log_path}")
    print(f"{'='*60}\n")

    # --- Cleanup ---
    del model, optimizer, scaler, scheduler, ema, noise_schedule
    del inception_extractor
    del train_loader, test_loader
    torch.cuda.empty_cache()
    print(f"[BASE/{activation.upper()}] VRAM liberata.")


# ==============================================================
# PLOT PER IL PAPER (Base-only, 5 plot)
# ==============================================================

def generate_plots():
    """Genera plot per il paper dai log JSON in results/."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results_dir = _get_results_dir()
    activations = ALL_ACTIVATIONS

    # --- Carica tutti i log ---
    def load_latest(pattern):
        files = sorted(glob.glob(os.path.join(results_dir, pattern)))
        if not files:
            return None
        with open(files[-1]) as f:
            return json.load(f)

    data = {}
    for act in activations:
        d = load_latest(f"dit_cifar10_base_{act}_*.json")
        if d:
            data[act] = d

    if not data:
        print("[PLOT] Nessun risultato trovato. Esegui prima gli esperimenti.")
        return

    colors = {"nova": "#E63946", "gelu": "#457B9D"}

    # ==========================================================
    # PLOT 1: Training + Val loss curves
    # ==========================================================
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DiT-Base DDPM CIFAR-10 — Training Curves (v2)",
                 fontsize=14, fontweight='bold')

    for act in activations:
        if act not in data:
            continue
        d = data[act]
        epochs_list = [e["epoch"] for e in d["epoche_log"]]
        train_losses = [e["train_loss"] for e in d["epoche_log"]]
        val_losses = [e["val_loss"] for e in d["epoche_log"]]
        best_val = d["metriche"]["best_val_loss"]

        ax_train.plot(epochs_list, train_losses,
                      color=colors.get(act, "#333"),
                      linewidth=1.5, label=f"{act.upper()} train")
        ax_val.plot(epochs_list, val_losses,
                    color=colors.get(act, "#333"),
                    linewidth=1.5,
                    label=f"{act.upper()} (best {best_val:.4f})")

    ax_train.set_title("Train Loss", fontsize=11)
    ax_train.set_xlabel("Epoca")
    ax_train.set_ylabel("MSE Loss")
    ax_train.legend(fontsize=9)
    ax_train.grid(True, alpha=0.3)

    ax_val.set_title("Val Loss (EMA)", fontsize=11)
    ax_val.set_xlabel("Epoca")
    ax_val.set_ylabel("MSE Loss")
    ax_val.legend(fontsize=9)
    ax_val.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "plot_dit_cifar10_v2_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # ==========================================================
    # PLOT 2: FID curves
    # ==========================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("DiT-Base DDPM CIFAR-10 — FID over Training (v2, DDIM)",
                 fontsize=13, fontweight='bold')

    for act in activations:
        if act not in data or not data[act].get("fid_is_log"):
            continue
        d = data[act]
        fid_epochs = [f["epoch"] for f in d["fid_is_log"]]
        fid_values = [f["fid"] for f in d["fid_is_log"]]
        best_fid = min(fid_values)
        ax.plot(fid_epochs, fid_values, color=colors.get(act, "#333"),
                marker='o', markersize=6, linewidth=2,
                label=f"{act.upper()} (best {best_fid:.1f})")

    ax.set_xlabel("Epoca", fontsize=11)
    ax.set_ylabel("FID (lower is better)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "plot_dit_cifar10_v2_fid_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # ==========================================================
    # PLOT 3: IS curves
    # ==========================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("DiT-Base DDPM CIFAR-10 — IS over Training (v2)",
                 fontsize=13, fontweight='bold')

    for act in activations:
        if act not in data or not data[act].get("fid_is_log"):
            continue
        d = data[act]
        is_epochs = [f["epoch"] for f in d["fid_is_log"]]
        is_values = [f["is_mean"] for f in d["fid_is_log"]]
        is_stds = [f["is_std"] for f in d["fid_is_log"]]
        best_is = max(is_values)
        ax.errorbar(is_epochs, is_values, yerr=is_stds,
                     color=colors.get(act, "#333"),
                     marker='s', markersize=6, linewidth=2, capsize=3,
                     label=f"{act.upper()} (best {best_is:.2f})")

    ax.set_xlabel("Epoca", fontsize=11)
    ax.set_ylabel("IS (higher is better)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "plot_dit_cifar10_v2_is_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # ==========================================================
    # PLOT 4: β evolution (NOVA only)
    # ==========================================================
    if "nova" in data:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Evoluzione del parametro β (NOVA) — DiT-Base v2",
                     fontsize=13, fontweight='bold')

        d = data["nova"]
        epochs_list = [e["epoch"] for e in d["epoche_log"]]
        betas = [e.get("beta", None) for e in d["epoche_log"]]
        if betas[0] is not None:
            ax.plot(epochs_list, betas, color=colors["nova"],
                    linewidth=2, label="β (NOVA)")

        ax.set_xlabel("Epoca", fontsize=11)
        ax.set_ylabel("β", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(results_dir,
                            "plot_dit_cifar10_v2_beta_evolution.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[PLOT] Salvato: {path}")

    # ==========================================================
    # PLOT 5: Sample grid comparison (ultima epoca)
    # ==========================================================
    from PIL import Image

    fig, axes = plt.subplots(1, len(activations),
                             figsize=(5 * len(activations), 5))
    if not hasattr(axes, '__len__'):
        axes = [axes]
    fig.suptitle("DiT-Base DDPM CIFAR-10 — Generated Samples (v2, final)",
                 fontsize=14, fontweight='bold')

    for idx, act in enumerate(activations):
        ax = axes[idx]
        if act not in data:
            ax.set_visible(False)
            continue

        d = data[act]
        last_epoch = d["epoche_log"][-1]["epoch"] if d["epoche_log"] else 400
        sample_pattern = os.path.join(
            results_dir,
            f"dit_cifar10_samples_base_{act}_epoch{last_epoch:03d}.png")
        sample_files = glob.glob(sample_pattern)

        if sample_files:
            img = Image.open(sample_files[0])
            ax.imshow(np.array(img))
            ax.set_title(f"Base / {act.upper()}", fontsize=11)
        else:
            ax.text(0.5, 0.5, "No samples", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f"Base / {act.upper()}", fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(results_dir, "plot_dit_cifar10_v2_sample_grid.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # ==========================================================
    # PLOT 6: Tabella riassuntiva v1 vs v2 (se dati v1 disponibili)
    # ==========================================================
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    ax.set_title("DiT-Base CIFAR-10 — NOVA vs GELU Summary (v2)",
                 fontsize=13, fontweight='bold', pad=20)

    table_data = []
    headers = ["Activation", "Best Val Loss", "Best FID ↓",
               "Best IS ↑", "Epochs"]
    for act in activations:
        if act not in data:
            continue
        d = data[act]
        m = d["metriche"]
        best_is = max(f["is_mean"] for f in d["fid_is_log"]) \
            if d.get("fid_is_log") else "N/A"
        table_data.append([
            act.upper(),
            f"{m['best_val_loss']:.6f}",
            f"{m.get('best_fid', 'N/A')}",
            f"{best_is}" if isinstance(best_is, str) else f"{best_is:.2f}",
            str(d["configurazione"]["epochs"]),
        ])

    if table_data:
        table = ax.table(cellText=table_data, colLabels=headers,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
        # Colora header
        for j, key in enumerate(headers):
            table[0, j].set_facecolor('#E8E8E8')
            table[0, j].set_text_props(fontweight='bold')

    plt.tight_layout()
    path = os.path.join(results_dir, "plot_dit_cifar10_v2_summary_table.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    print(f"\n[PLOT] Tutti i plot generati in: {results_dir}")


# ==============================================================
# LAUNCHER: parallelo su 2 GPU
# ==============================================================

def launch_all(activations=None):
    global _nova_cuda_ext

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

    # 3. Lancia le attivazioni a coppie sulle 2 GPU
    script = os.path.abspath(__file__)
    cfg = BASE_CONFIG

    print(f"\n{'#'*60}")
    print(f"  SCALA: BASE ({cfg['num_layers']}L, "
          f"{cfg['embed_dim']}d, {cfg['num_heads']}h)")
    print(f"  Batch: {cfg['batch_size']}, LR: {cfg['lr']}")
    print(f"{'#'*60}")

    pairs = [(activations[i],
              activations[i + 1] if i + 1 < len(activations) else None)
             for i in range(0, len(activations), 2)]

    for pair in pairs:
        procs = []
        for gpu, act in enumerate(pair):
            if act is None:
                continue
            cmd = [sys.executable, script,
                   "--activation", act,
                   "--gpu", str(gpu)]
            print(f"\n[LAUNCHER] Avvio: {' '.join(cmd)}")
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
            procs.append((act, p))

        for act, p in procs:
            p.wait()
            if p.returncode != 0:
                print(f"[LAUNCHER] ERRORE: base/{act} "
                      f"terminato con codice {p.returncode}")
            else:
                print(f"[LAUNCHER] BASE/{act.upper()} "
                      f"completato con successo.")

    # 4. Genera i plot
    print("\n[LAUNCHER] Generazione plot per il paper...")
    generate_plots()

    print(f"\n{'='*60}")
    print("  DiT DDPM v2 CIFAR-10 COMPLETATO")
    print(f"{'='*60}")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DiT DDPM v2 su CIFAR-10: NOVA vs GELU (solo Base)")
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
    elif args.activation is not None:
        run_experiment(args.activation, args.gpu)
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            print(f"[ATTENZIONE] Solo {n_gpus} GPU disponibili. "
                  "Lancia manualmente con --activation e --gpu.")
            sys.exit(1)

        launch_all()
