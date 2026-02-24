#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESPERIMENTO: Sensibilità a β₀ v2 — Validazione cross-scala + training esteso
==============================================================================
Estende nova_beta_sensitivity.py (v1, solo ViT-Small) con:

  A) Cross-scala: β₀ ∈ {0.5, 1.0, 1.5, 2.0} su ViT-Tiny e ViT-Base
     (8 run, 100 epoche) per verificare che il plateau β₀ ∈ [1.0, 2.0]
     sia indipendente dalla scala del modello.

  B) Training esteso: β₀ ∈ {0.1, 3.0} su ViT-Small a 200 epoche
     (2 run) per verificare se il "danno precoce" delle β₀ estreme è
     irreversibile o semplicemente una convergenza più lenta.

Architettura e regolarizzazione identiche a vit_scaling_v2.py (DeiT-style).

Uso:
    # Lancia tutto (8 cross-scala + 2 estesi, a coppie su 2 GPU):
    python nova_beta_sensitivity_v2.py

    # Singolo esperimento:
    python nova_beta_sensitivity_v2.py --scale tiny --beta-init 1.0 --gpu 0
    python nova_beta_sensitivity_v2.py --scale small --beta-init 0.1 --epochs 200 --gpu 0

    # Solo generazione plot (combina risultati v1 e v2):
    python nova_beta_sensitivity_v2.py --plot-only
"""

import argparse
import subprocess
import sys
import os
import time
import json
import glob
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2

# ==============================================================
# CONFIGURAZIONE ESPERIMENTO
# ==============================================================
SEED = 42
DATASET = "CIFAR-100"
HARDWARE = "NVIDIA T4"
NUM_WORKERS = 4
PATCH_SIZE = 4
MLP_RATIO = 4
NUM_CLASSES = 100
IMG_SIZE = 32
LABEL_SMOOTHING = 0.1
MAX_GRAD_NORM = 1.0

# Configurazioni per scala (identiche a vit_scaling_v2.py)
SCALING_CONFIGS = {
    "tiny": {
        "embed_dim": 256, "num_heads": 4, "num_layers": 4,
        "batch_size": 1024, "lr": 3e-3, "warmup_epochs": 10,
        "dropout": 0.1, "weight_decay": 0.05, "drop_path_rate": 0.1,
        "cutmix_alpha": 1.0, "mixup_alpha": 0.8,
        "cutmix_mixup_prob": 1.0, "switch_prob": 0.5,
        "randaug_num_ops": 2, "randaug_magnitude": 9,
    },
    "small": {
        "embed_dim": 384, "num_heads": 6, "num_layers": 6,
        "batch_size": 512, "lr": 1e-3, "warmup_epochs": 15,
        "dropout": 0.1, "weight_decay": 0.05, "drop_path_rate": 0.2,
        "cutmix_alpha": 1.0, "mixup_alpha": 0.8,
        "cutmix_mixup_prob": 1.0, "switch_prob": 0.5,
        "randaug_num_ops": 2, "randaug_magnitude": 9,
    },
    "base": {
        "embed_dim": 512, "num_heads": 8, "num_layers": 8,
        "batch_size": 256, "lr": 5e-4, "warmup_epochs": 20,
        "dropout": 0.1, "weight_decay": 0.05, "drop_path_rate": 0.3,
        "cutmix_alpha": 1.0, "mixup_alpha": 0.8,
        "cutmix_mixup_prob": 1.0, "switch_prob": 0.5,
        "randaug_num_ops": 2, "randaug_magnitude": 9,
    },
}

# Piano esperimenti:
# A) Cross-scala: Tiny + Base, β₀ ∈ {0.5, 1.0, 1.5, 2.0}, 100 epoche
CROSS_SCALE_BETAS = [0.5, 1.0, 1.5, 2.0]
CROSS_SCALE_EPOCHS = 100
# B) Training esteso: Small, β₀ ∈ {0.1, 3.0}, 200 epoche
EXTENDED_BETAS = [0.1, 3.0]
EXTENDED_EPOCHS = 200


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
        name='nova_cuda_ext_beta_v2',
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


# ==============================================================
# STOCHASTIC DEPTH (DropPath)
# ==============================================================

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


# ==============================================================
# MODELLO: Vision Transformer (identico a vit_scaling_v2.py)
# ==============================================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout, act_layer,
                 drop_path_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ScalableViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes,
                 embed_dim, num_heads, num_layers, mlp_ratio, dropout,
                 act_layer, drop_path_rate=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout,
                             act_layer, drop_path_rate=dpr[i])
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
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
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x[:, 0])
        return self.head(x)


# ==============================================================
# DATI: CIFAR-100 con RandAugment
# ==============================================================

def get_cifar100_loaders(batch_size, num_workers, randaug_num_ops=2,
                         randaug_magnitude=9):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=randaug_num_ops,
                               magnitude=randaug_magnitude),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_root = _get_data_root()
    train_set = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


# ==============================================================
# CUTMIX + MIXUP
# ==============================================================

class CutMixMixupCollator:
    def __init__(self, num_classes, cutmix_alpha=1.0, mixup_alpha=0.8,
                 prob=1.0, switch_prob=0.5):
        self.cutmix = transforms_v2.CutMix(
            alpha=cutmix_alpha, num_classes=num_classes)
        self.mixup = transforms_v2.MixUp(
            alpha=mixup_alpha, num_classes=num_classes)
        self.prob = prob
        self.switch_prob = switch_prob

    def __call__(self, images, labels):
        if random.random() > self.prob:
            return images, labels
        if random.random() < self.switch_prob:
            return self.cutmix(images, labels)
        return self.mixup(images, labels)


# ==============================================================
# TRAINING & VALUTAZIONE
# ==============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    cutmix_mixup=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        mixed = False
        if cutmix_mixup is not None:
            images, labels = cutmix_mixup(images, labels)
            mixed = True

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        _, predicted = outputs.max(1)
        if not mixed:
            correct += predicted.eq(labels).sum().item()
        else:
            if labels.ndim > 1:
                _, target_max = labels.max(1)
                correct += predicted.eq(target_max).sum().item()
            else:
                correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def build_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0,
        total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ==============================================================
# ESPERIMENTO PRINCIPALE
# ==============================================================

def run_experiment(scale: str, beta_init: float, epochs: int,
                   gpu: int) -> None:
    set_seed(SEED)

    cfg = SCALING_CONFIGS[scale]
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    embed_dim = cfg["embed_dim"]
    num_heads = cfg["num_heads"]
    num_layers = cfg["num_layers"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    warmup_epochs = cfg["warmup_epochs"]
    dropout = cfg["dropout"]
    weight_decay = cfg["weight_decay"]
    drop_path_rate = cfg["drop_path_rate"]

    beta_str = f"{beta_init:.2f}".replace(".", "p")
    is_extended = epochs > CROSS_SCALE_EPOCHS
    tag = "ext" if is_extended else "xs"
    experiment_name = f"nova_beta_v2_{scale}_{tag}_b{beta_str}"
    model_desc = (f"ViT-{scale.capitalize()} ({num_layers}L, {embed_dim}d, "
                  f"{num_heads}h, MLP×{MLP_RATIO}) + NOVA(β₀={beta_init})")

    print(f"\n{'='*60}")
    print(f"  ESPERIMENTO: {experiment_name}")
    print(f"  GPU: {gpu} ({torch.cuda.get_device_name(device)})")
    print(f"  Scala: {scale.upper()} | β₀ = {beta_init} | Epoche: {epochs}")
    print(f"  Tipo: {'Training esteso' if is_extended else 'Cross-scala'}")
    print(f"  Architettura: {num_layers}L, {embed_dim}d, {num_heads}h")
    print(f"  Batch: {batch_size}, LR: {lr}")
    print(f"  DropPath: {drop_path_rate}, RandAug+CutMix+Mixup")
    print(f"  Mixed Precision: FP16 | Seed: {SEED}")
    print(f"{'='*60}\n")

    # --- Activation ---
    act_layer, nova_backend = make_nova(beta=beta_init)
    print(f"[INFO] NOVA backend: {nova_backend}, β₀ = {beta_init}")

    # --- Modello ---
    model = ScalableViT(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=3,
        num_classes=NUM_CLASSES, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_layers, mlp_ratio=MLP_RATIO, dropout=dropout,
        act_layer=act_layer, drop_path_rate=drop_path_rate,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parametri totali: {num_params:,}")

    # --- Dati ---
    train_loader, test_loader = get_cifar100_loaders(
        batch_size, NUM_WORKERS,
        randaug_num_ops=cfg["randaug_num_ops"],
        randaug_magnitude=cfg["randaug_magnitude"],
    )

    # --- CutMix/Mixup ---
    cutmix_mixup = CutMixMixupCollator(
        num_classes=NUM_CLASSES,
        cutmix_alpha=cfg["cutmix_alpha"],
        mixup_alpha=cfg["mixup_alpha"],
        prob=cfg["cutmix_mixup_prob"],
        switch_prob=cfg["switch_prob"],
    )

    # --- Ottimizzatore, Loss, Scheduler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
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
        "obiettivo": (f"Beta sensitivity v2 ({'esteso' if is_extended else 'cross-scala'}): "
                      f"{scale}, β₀={beta_init}, {epochs}ep"),
        "configurazione": {
            "scale": scale,
            "beta_init": beta_init,
            "epochs": epochs,
            "experiment_type": "extended" if is_extended else "cross_scale",
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "patch_size": PATCH_SIZE,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "mlp_ratio": MLP_RATIO,
            "dropout": dropout,
            "drop_path_rate": drop_path_rate,
            "label_smoothing": LABEL_SMOOTHING,
            "mixed_precision": "fp16",
            "num_params": num_params,
            "nova_backend": nova_backend,
            "randaug_num_ops": cfg["randaug_num_ops"],
            "randaug_magnitude": cfg["randaug_magnitude"],
            "cutmix_alpha": cfg["cutmix_alpha"],
            "mixup_alpha": cfg["mixup_alpha"],
        },
        "metriche": {},
        "epoche_log": [],
        "tempo_totale_sec": None,
    }

    # --- Training loop ---
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            cutmix_mixup=cutmix_mixup)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        beta_val = None
        for m in model.modules():
            if isinstance(m, (NOVACuda, NOVAPython)):
                beta_val = round(m.beta.item(), 6)
                break

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "lr": round(current_lr, 8),
            "beta": beta_val,
            "epoch_time_sec": round(epoch_time, 1),
        }
        log_data["epoche_log"].append(epoch_log)

        # Salvataggio incrementale
        log_data["metriche"] = {
            "best_val_acc": round(best_val_acc, 2),
            "final_val_acc": round(val_acc, 2),
            "final_train_loss": round(train_loss, 4),
            "final_val_loss": round(val_loss, 4),
            "final_train_acc": round(train_acc, 2),
            "final_beta": beta_val,
            "beta_init": beta_init,
        }
        log_data["tempo_totale_sec"] = round(time.time() - start_time, 2)
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        print(f"[{scale.upper()}/β₀={beta_init}/{tag}] "
              f"Epoch {epoch+1:03d}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_acc={val_acc:.2f}%  "
              f"β={beta_val}  lr={current_lr:.6f}  ({epoch_time:.1f}s)")

    elapsed = time.time() - start_time

    log_data["tempo_totale_sec"] = round(elapsed, 2)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    print(f"\n{'='*60}")
    print(f"  RISULTATI — {scale.upper()} / β₀={beta_init} / {epochs}ep")
    print(f"  Parametri: {num_params:,}")
    print(f"  Best val acc: {best_val_acc:.2f}%")
    print(f"  β finale: {beta_val}")
    print(f"  Tempo totale: {elapsed:.1f}s")
    print(f"  Log: {log_path}")
    print(f"{'='*60}\n")

    del model, optimizer, scaler, scheduler, criterion
    del train_loader, test_loader
    torch.cuda.empty_cache()


# ==============================================================
# PLOT
# ==============================================================

def _load_all_beta_logs(results_dir):
    """Carica tutti i log beta sensitivity (v1 + v2)."""
    logs = {}  # (scale, beta_init, experiment_type) -> log_data

    for f in sorted(os.listdir(results_dir)):
        if not f.endswith(".json"):
            continue

        # v1: nova_beta_sensitivity_b{X}_{ts}.json (scale=small, type=v1)
        if f.startswith("nova_beta_sensitivity_b") and not f.startswith("nova_beta_v2"):
            path = os.path.join(results_dir, f)
            with open(path) as fh:
                data = json.load(fh)
            b0 = data["configurazione"]["beta_init"]
            logs[("small", b0, "v1")] = data

        # v2: nova_beta_v2_{scale}_{tag}_b{X}_{ts}.json
        elif f.startswith("nova_beta_v2_"):
            path = os.path.join(results_dir, f)
            with open(path) as fh:
                data = json.load(fh)
            cfg = data["configurazione"]
            scale = cfg["scale"]
            b0 = cfg["beta_init"]
            etype = cfg.get("experiment_type", "cross_scale")
            logs[(scale, b0, etype)] = data

    return logs


def generate_plots():
    """Genera plot combinati v1 + v2."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
    except ImportError:
        print("[PLOT] matplotlib non disponibile.")
        return

    results_dir = _get_results_dir()
    logs = _load_all_beta_logs(results_dir)

    if not logs:
        print("[PLOT] Nessun log beta sensitivity trovato.")
        return

    # Organizza per scala
    scales_found = sorted(set(k[0] for k in logs))
    print(f"[PLOT] Scale trovate: {scales_found}")
    for s in scales_found:
        betas = sorted(set(k[1] for k in logs if k[0] == s))
        print(f"  {s}: β₀ = {betas}")

    scale_colors = {"tiny": "#264653", "small": "#2A9D8F", "base": "#E76F51"}
    scale_labels = {"tiny": "Tiny (3.2M)", "small": "Small (10.7M)",
                    "base": "Base (25.3M)"}

    # ===========================================================
    # PLOT 1: β convergenza — un pannello per scala
    # ===========================================================
    n_scales = len(scales_found)
    fig, axes = plt.subplots(1, n_scales, figsize=(6 * n_scales, 5))
    if n_scales == 1:
        axes = [axes]
    fig.suptitle("Convergenza di β per diversi β₀ — per scala",
                 fontsize=14, fontweight='bold')

    for ax, scale in zip(axes, scales_found):
        betas_for_scale = sorted(set(
            k[1] for k in logs if k[0] == scale))

        cmap = get_cmap("coolwarm")
        b_min = min(betas_for_scale) if betas_for_scale else 0
        b_max = max(betas_for_scale) if betas_for_scale else 1

        for b0 in betas_for_scale:
            # Prendi il log con più epoche (extended > v1 > cross_scale)
            candidates = [(k, v) for k, v in logs.items()
                          if k[0] == scale and k[1] == b0]
            if not candidates:
                continue
            data = max(candidates, key=lambda x: len(x[1]["epoche_log"]))[1]

            epochs = [e["epoch"] for e in data["epoche_log"]]
            bvals = [e["beta"] for e in data["epoche_log"]]
            n_ep = len(epochs)
            t = (b0 - b_min) / (b_max - b_min) if b_max > b_min else 0.5
            label = f"β₀={b0}" + (f" ({n_ep}ep)" if n_ep > 100 else "")
            ax.plot(epochs, bvals, color=cmap(t), linewidth=2.0, label=label)

        ax.set_title(scale_labels.get(scale, scale), fontsize=12)
        ax.set_xlabel("Epoca")
        ax.set_ylabel("β")
        ax.legend(fontsize=8, ncol=2, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "plot_beta_v2_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # ===========================================================
    # PLOT 2: Accuracy vs β₀ per scala (curve sovrapposte)
    # ===========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Best Val Accuracy vs β₀ — per scala (ViT v2, CIFAR-100)",
                 fontsize=13, fontweight='bold')

    for scale in scales_found:
        # Per ogni scala, raccogli (β₀, best_acc) usando le run a 100ep
        points = []
        for (s, b0, etype), data in logs.items():
            if s != scale:
                continue
            # Per cross-scala e v1 usa direttamente; per extended usa
            # l'accuracy a epoca 100 se disponibile, altrimenti best
            if etype == "extended":
                # Prendi accuracy a epoch 100 per confronto equo
                ep100 = [e for e in data["epoche_log"] if e["epoch"] == 100]
                if ep100:
                    acc = max(e["val_acc"] for e in data["epoche_log"]
                              if e["epoch"] <= 100)
                else:
                    acc = data["metriche"]["best_val_acc"]
            else:
                acc = data["metriche"]["best_val_acc"]
            points.append((b0, acc))

        if not points:
            continue
        points.sort()
        betas_x = [p[0] for p in points]
        accs_y = [p[1] for p in points]

        ax.plot(betas_x, accs_y, color=scale_colors[scale],
                marker='o', markersize=8, linewidth=2.5,
                label=scale_labels.get(scale, scale))
        for bx, ay in zip(betas_x, accs_y):
            ax.annotate(f'{ay:.1f}', (bx, ay), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8,
                        color=scale_colors[scale], fontweight='bold')

    ax.set_xlabel("β₀ (inizializzazione)", fontsize=11)
    ax.set_ylabel("Best Val Accuracy (%) a 100 epoche", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "plot_beta_v2_accuracy_vs_b0.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # ===========================================================
    # PLOT 3: Training esteso — β₀ estreme, 100 vs 200 epoche
    # ===========================================================
    extended_logs = {(k[0], k[1]): v for k, v in logs.items()
                     if k[2] == "extended"}
    v1_logs = {(k[0], k[1]): v for k, v in logs.items()
               if k[2] == "v1"}

    if extended_logs:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Training Esteso (200ep) vs Standard (100ep) — β₀ estreme",
                     fontsize=13, fontweight='bold')

        # Pannello sinistro: val accuracy
        ax = axes[0]
        for (scale, b0), data in sorted(extended_logs.items()):
            epochs = [e["epoch"] for e in data["epoche_log"]]
            val_accs = [e["val_acc"] for e in data["epoche_log"]]
            color = "#1D3557" if b0 < 1.0 else "#E63946"
            ax.plot(epochs, val_accs, color=color, linewidth=2.0,
                    label=f"β₀={b0} (200ep)")

            # Sovrapponi la v1 (100ep) se disponibile
            v1_key = (scale, b0)
            if v1_key in v1_logs:
                v1_data = v1_logs[v1_key]
                v1_epochs = [e["epoch"] for e in v1_data["epoche_log"]]
                v1_accs = [e["val_acc"] for e in v1_data["epoche_log"]]
                ax.plot(v1_epochs, v1_accs, color=color, linewidth=1.5,
                        linestyle='--', alpha=0.6, label=f"β₀={b0} (100ep)")

        ax.axvline(x=100, color='gray', linestyle=':', alpha=0.5,
                   label='Limite 100ep')
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Val Accuracy (%)")
        ax.set_title("Val Accuracy")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Pannello destro: evoluzione β
        ax = axes[1]
        for (scale, b0), data in sorted(extended_logs.items()):
            epochs = [e["epoch"] for e in data["epoche_log"]]
            bvals = [e["beta"] for e in data["epoche_log"]]
            color = "#1D3557" if b0 < 1.0 else "#E63946"
            ax.plot(epochs, bvals, color=color, linewidth=2.0,
                    label=f"β₀={b0} (200ep)")

            v1_key = (scale, b0)
            if v1_key in v1_logs:
                v1_data = v1_logs[v1_key]
                v1_epochs = [e["epoch"] for e in v1_data["epoche_log"]]
                v1_betas = [e["beta"] for e in v1_data["epoche_log"]]
                ax.plot(v1_epochs, v1_betas, color=color, linewidth=1.5,
                        linestyle='--', alpha=0.6, label=f"β₀={b0} (100ep)")

        ax.axvline(x=100, color='gray', linestyle=':', alpha=0.5,
                   label='Limite 100ep')
        # Linea di riferimento: β* ≈ 0.45 (bacino attrattore da v1)
        ax.axhline(y=0.45, color='green', linestyle='--', alpha=0.4,
                   label='β* ≈ 0.45 (bacino)')
        ax.set_xlabel("Epoca")
        ax.set_ylabel("β")
        ax.set_title("Evoluzione β")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(results_dir, "plot_beta_v2_extended.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[PLOT] Salvato: {path}")

    # ===========================================================
    # PLOT 4: Heatmap β₀ × scala → accuracy
    # ===========================================================
    all_betas = sorted(set(k[1] for k in logs))
    all_scales = ["tiny", "small", "base"]
    scales_present = [s for s in all_scales if s in scales_found]

    if len(scales_present) >= 2:
        fig, ax = plt.subplots(figsize=(max(8, len(all_betas) * 1.2),
                                        len(scales_present) * 1.5 + 2))
        ax.set_title("Best Val Accuracy: β₀ × Scala (100ep)",
                      fontsize=13, fontweight='bold')

        matrix = np.full((len(scales_present), len(all_betas)), np.nan)
        for i, scale in enumerate(scales_present):
            for j, b0 in enumerate(all_betas):
                candidates = [(k, v) for k, v in logs.items()
                              if k[0] == scale and k[1] == b0]
                if not candidates:
                    continue
                # Usa accuracy a 100 epoche per confronto equo
                data = candidates[0][1]
                if data["configurazione"].get("experiment_type") == "extended":
                    ep100 = [e for e in data["epoche_log"] if e["epoch"] <= 100]
                    if ep100:
                        matrix[i, j] = max(e["val_acc"] for e in ep100)
                else:
                    matrix[i, j] = data["metriche"]["best_val_acc"]

        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                        vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
        for i in range(len(scales_present)):
            for j in range(len(all_betas)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                            fontsize=10, fontweight='bold',
                            color='white' if val < np.nanmedian(matrix) else 'black')

        ax.set_xticks(range(len(all_betas)))
        ax.set_xticklabels([f'{b}' for b in all_betas])
        ax.set_yticks(range(len(scales_present)))
        ax.set_yticklabels([scale_labels.get(s, s) for s in scales_present])
        ax.set_xlabel("β₀", fontsize=11)
        fig.colorbar(im, ax=ax, label="Best Val Accuracy (%)", shrink=0.8)
        plt.tight_layout()
        path = os.path.join(results_dir, "plot_beta_v2_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[PLOT] Salvato: {path}")

    # ===========================================================
    # TABELLA RIASSUNTIVA
    # ===========================================================
    print(f"\n{'='*80}")
    print(f"  RIEPILOGO BETA SENSITIVITY v2")
    print(f"{'='*80}")
    print(f"  {'Scala':<8} {'β₀':>5}  {'Epoche':>6}  {'Best Acc':>10}  "
          f"{'β finale':>10}  {'Tipo':<12}")
    print(f"  {'-'*60}")

    for key in sorted(logs.keys()):
        scale, b0, etype = key
        data = logs[key]
        m = data["metriche"]
        n_ep = len(data["epoche_log"])
        print(f"  {scale:<8} {b0:>5.2f}  {n_ep:>6}  {m['best_val_acc']:>9.2f}%  "
              f"{m.get('final_beta', '?'):>10}  {etype:<12}")

    print(f"{'='*80}\n")


# ==============================================================
# LAUNCHER
# ==============================================================

def launch_all():
    global _nova_cuda_ext

    # 1. Pre-download dataset
    print("[LAUNCHER] Pre-download dataset CIFAR-100...")
    data_root = _get_data_root()
    torchvision.datasets.CIFAR100(root=data_root, train=True, download=True)
    torchvision.datasets.CIFAR100(root=data_root, train=False, download=True)
    print("[LAUNCHER] Dataset pronto.")

    # 2. Pre-compilazione kernel CUDA
    print("[LAUNCHER] Pre-compilazione kernel CUDA NOVA (30-60s)...")
    try:
        _nova_cuda_ext = _compile_nova_cuda()
        print("[LAUNCHER] Kernel CUDA compilato e cachato.")
    except Exception as e:
        print(f"[LAUNCHER] ATTENZIONE: Compilazione CUDA fallita: {e}")

    script = os.path.abspath(__file__)

    # ==========================================================
    # A) CROSS-SCALA: Tiny + Base, β₀ ∈ {0.5, 1.0, 1.5, 2.0}
    # ==========================================================
    print(f"\n{'#'*60}")
    print(f"  FASE A: CROSS-SCALA (Tiny + Base, 100 epoche)")
    print(f"{'#'*60}")

    for scale in ["tiny", "base"]:
        # Lancia β₀ a coppie su 2 GPU
        pairs = []
        for i in range(0, len(CROSS_SCALE_BETAS), 2):
            pair = [CROSS_SCALE_BETAS[i]]
            if i + 1 < len(CROSS_SCALE_BETAS):
                pair.append(CROSS_SCALE_BETAS[i + 1])
            pairs.append(pair)

        for pair in pairs:
            procs = []
            for gpu, b0 in enumerate(pair):
                cmd = [sys.executable, script,
                       "--scale", scale,
                       "--beta-init", str(b0),
                       "--epochs", str(CROSS_SCALE_EPOCHS),
                       "--gpu", str(gpu)]
                print(f"\n[LAUNCHER] Avvio: {' '.join(cmd)}")
                p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
                procs.append((f"{scale}/β₀={b0}", p))

            for name, p in procs:
                p.wait()
                rc = "OK" if p.returncode == 0 else f"ERRORE ({p.returncode})"
                print(f"[LAUNCHER] {name}: {rc}")

    # ==========================================================
    # B) TRAINING ESTESO: Small, β₀ ∈ {0.1, 3.0}, 200 epoche
    # ==========================================================
    print(f"\n{'#'*60}")
    print(f"  FASE B: TRAINING ESTESO (Small, 200 epoche)")
    print(f"{'#'*60}")

    procs = []
    for gpu, b0 in enumerate(EXTENDED_BETAS):
        cmd = [sys.executable, script,
               "--scale", "small",
               "--beta-init", str(b0),
               "--epochs", str(EXTENDED_EPOCHS),
               "--gpu", str(gpu)]
        print(f"\n[LAUNCHER] Avvio: {' '.join(cmd)}")
        p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        procs.append((f"small/β₀={b0}/200ep", p))

    for name, p in procs:
        p.wait()
        rc = "OK" if p.returncode == 0 else f"ERRORE ({p.returncode})"
        print(f"[LAUNCHER] {name}: {rc}")

    # 3. Plot
    print("\n[LAUNCHER] Generazione plot combinati v1+v2...")
    generate_plots()

    print(f"\n{'='*60}")
    print("  BETA SENSITIVITY v2 COMPLETATO")
    print(f"{'='*60}")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NOVA β₀ sensitivity v2: cross-scala + training esteso")
    parser.add_argument("--scale", type=str,
                        choices=list(SCALING_CONFIGS.keys()),
                        help="Scala del modello")
    parser.add_argument("--beta-init", type=float,
                        help="Valore di β₀")
    parser.add_argument("--epochs", type=int, default=CROSS_SCALE_EPOCHS,
                        help=f"Numero di epoche (default: {CROSS_SCALE_EPOCHS})")
    parser.add_argument("--gpu", type=int, default=0,
                        help="ID della GPU (default: 0)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Solo generazione plot dai log esistenti")
    args = parser.parse_args()

    if args.plot_only:
        generate_plots()
    elif args.scale is not None and args.beta_init is not None:
        run_experiment(args.scale, args.beta_init, args.epochs, args.gpu)
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            print(f"[ATTENZIONE] Solo {n_gpus} GPU disponibili.")
            print("Lancia manualmente con --scale, --beta-init, --epochs e --gpu.")
            sys.exit(1)
        launch_all()
