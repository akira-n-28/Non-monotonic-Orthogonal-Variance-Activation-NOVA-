#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESPERIMENTO: ConvNeXt Scaling su CIFAR-100
============================================
Valida il vantaggio di NOVA su architetture pure convolutive (no attention).

ConvNeXt (Liu et al. 2022) usa GELU per design — sostituirla con NOVA è un
test diretto e pulito dell'impatto della funzione di attivazione su CNN.

Architettura: ConvNeXt adattato per CIFAR-100 (32×32) con stem stride-1.
3 scale calibrate per avere parametri comparabili ai ViT:
  - Tiny:  dims [40, 80, 160, 320],  depths [2,2,6,2]  (~3.4M params)
  - Small: dims [64, 128, 256, 512], depths [3,3,6,3]  (~10.7M params)
  - Base:  dims [96, 192, 384, 768], depths [3,3,9,3]  (~28M params)

Uso:
    # Lancia tutto (tutte le scale, NOVA vs GELU su 2 GPU):
    python convnext_cifar100.py

    # Singolo esperimento:
    python convnext_cifar100.py --scale tiny --activation nova --gpu 0

    # Solo generazione plot:
    python convnext_cifar100.py --plot-only
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
IMG_SIZE = 32
NUM_CLASSES = 100
LABEL_SMOOTHING = 0.1
MAX_GRAD_NORM = 1.0
LAYER_SCALE_INIT = 1e-6

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


# Scale ConvNeXt calibrate per avere parametri comparabili ai ViT.
# Batch size da calibrare su T4 15 GB (ConvNeXt è più efficiente del ViT).
SCALING_CONFIGS = {
    "tiny": {
        "dims": [40, 80, 160, 320],
        "depths": [2, 2, 6, 2],
        "batch_size": 2048,
        "lr": 4e-3,
        "epochs": 100,
        "warmup_epochs": 10,
        "weight_decay": 0.05,
        "drop_path_rate": 0.05,
        "cutmix_alpha": 1.0,
        "mixup_alpha": 0.8,
        "cutmix_mixup_prob": 0.5,
        "switch_prob": 0.5,
        "randaug_num_ops": 2,
        "randaug_magnitude": 5,
    },
    "small": {
        "dims": [64, 128, 256, 512],
        "depths": [3, 3, 6, 3],
        "batch_size": 1024,
        "lr": 3e-3,
        "epochs": 100,
        "warmup_epochs": 15,
        "weight_decay": 0.05,
        "drop_path_rate": 0.2,
        "cutmix_alpha": 1.0,
        "mixup_alpha": 0.8,
        "cutmix_mixup_prob": 1.0,
        "switch_prob": 0.5,
        "randaug_num_ops": 2,
        "randaug_magnitude": 9,
    },
    "base": {
        "dims": [96, 192, 384, 768],
        "depths": [3, 3, 9, 3],
        "batch_size": 512,
        "lr": 2e-3,
        "epochs": 100,
        "warmup_epochs": 20,
        "weight_decay": 0.05,
        "drop_path_rate": 0.3,
        "cutmix_alpha": 1.0,
        "mixup_alpha": 0.8,
        "cutmix_mixup_prob": 1.0,
        "switch_prob": 0.5,
        "randaug_num_ops": 2,
        "randaug_magnitude": 9,
    },
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
        name='nova_cuda_ext_cnx',
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
# MODELLO: ConvNeXt per CIFAR-100
# ==============================================================

class ConvNeXtBlock(nn.Module):
    """Blocco ConvNeXt: DWConv 7×7 → LN → Linear → Act → Linear → LayerScale."""
    def __init__(self, dim, act_layer, drop_path_rate=0.0,
                 layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = act_layer
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = (DropPath(drop_path_rate)
                          if drop_path_rate > 0.0 else nn.Identity())

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)       # BCHW → BHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)       # BHWC → BCHW
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt adattato per CIFAR-100 (32×32).

    Stem: Conv 3×3 stride 1 (mantiene 32×32).
    4 stage con downsample LN + Conv 2×2 stride 2 tra gli stage.
    Risoluzione: 32 → 16 → 8 → 4.
    """
    def __init__(self, in_channels, num_classes, dims, depths,
                 act_layer, drop_path_rate=0.0):
        super().__init__()
        # --- Stem: Conv 3×3 stride 1, pad 1 ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([dims[0], IMG_SIZE, IMG_SIZE]),
        )

        # --- Stochastic depth rates per blocco ---
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        block_idx = 0

        # --- 4 Stage ---
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(4):
            # Downsample tra stage (tranne il primo)
            if i > 0:
                ds = nn.Sequential(
                    # LN su canali (permute necessario)
                    _ChannelLastLayerNorm(dims[i - 1]),
                    nn.Conv2d(dims[i - 1], dims[i],
                              kernel_size=2, stride=2),
                )
                self.downsamples.append(ds)
            else:
                self.downsamples.append(nn.Identity())

            # Blocchi dello stage
            blocks = []
            for j in range(depths[i]):
                blocks.append(ConvNeXtBlock(
                    dim=dims[i],
                    act_layer=act_layer,
                    drop_path_rate=dpr[block_idx],
                    layer_scale_init=LAYER_SCALE_INIT,
                ))
                block_idx += 1
            self.stages.append(nn.Sequential(*blocks))

        # --- Head ---
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for ds, stage in zip(self.downsamples, self.stages):
            x = ds(x)
            x = stage(x)
        # Global average pooling
        x = x.mean(dim=[-2, -1])     # BCHW → BC
        x = self.norm(x)
        return self.head(x)


class _ChannelLastLayerNorm(nn.Module):
    """LayerNorm applicato sui canali per tensor BCHW."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # BCHW → BHWC → LN → BHWC → BCHW
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


# ==============================================================
# DATI: CIFAR-100
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
        if mixed and labels.ndim > 1:
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

def run_experiment(scale: str, activation: str, gpu: int) -> None:
    set_seed(SEED)

    cfg = SCALING_CONFIGS[scale]
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    dims = cfg["dims"]
    depths = cfg["depths"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    epochs = cfg["epochs"]
    warmup_epochs = cfg["warmup_epochs"]
    weight_decay = cfg["weight_decay"]
    drop_path_rate = cfg["drop_path_rate"]

    experiment_name = f"convnext_cifar100_{scale}_{activation}"
    model_desc = (f"ConvNeXt-{scale.capitalize()} "
                  f"(dims={dims}, depths={depths}) + {activation.upper()}")

    print(f"\n{'='*60}")
    print(f"  ESPERIMENTO: {experiment_name}")
    print(f"  Dataset: {DATASET} (100 classi, 32×32)")
    print(f"  GPU: {gpu} ({torch.cuda.get_device_name(device)})")
    print(f"  Scala: {scale.upper()} | Attivazione: {activation.upper()}")
    print(f"  Architettura: dims={dims}, depths={depths}")
    print(f"  Epoche: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  DropPath: {drop_path_rate}, WD: {weight_decay}")
    print(f"  RandAugment: ops={cfg['randaug_num_ops']}, mag={cfg['randaug_magnitude']}")
    print(f"  CutMix/Mixup prob: {cfg['cutmix_mixup_prob']}")
    print(f"  Mixed Precision: FP16 | Seed: {SEED}")
    print(f"{'='*60}\n")

    # --- Activation layer ---
    nova_backend = None
    if activation == "nova":
        act_layer, nova_backend = make_nova(beta=1.0)
        print(f"[INFO] NOVA backend: {nova_backend}")
    elif activation == "gelu":
        act_layer = nn.GELU()
    elif activation == "silu":
        act_layer = nn.SiLU()
    elif activation == "mish":
        act_layer = nn.Mish()
    elif activation == "relu":
        act_layer = nn.ReLU()
    else:
        raise ValueError(f"Attivazione non supportata: {activation}")

    # --- Modello ---
    model = ConvNeXt(
        in_channels=3, num_classes=NUM_CLASSES,
        dims=dims, depths=depths,
        act_layer=act_layer,
        drop_path_rate=drop_path_rate,
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
        "obiettivo": (f"ConvNeXt scaling su CIFAR-100: {scale}, "
                      f"{activation} a {epochs} epoche"),
        "configurazione": {
            "scale": scale,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "img_size": IMG_SIZE,
            "num_classes": NUM_CLASSES,
            "dims": dims,
            "depths": depths,
            "drop_path_rate": drop_path_rate,
            "label_smoothing": LABEL_SMOOTHING,
            "layer_scale_init": LAYER_SCALE_INIT,
            "mixed_precision": "fp16",
            "num_params": num_params,
            "randaug_num_ops": cfg["randaug_num_ops"],
            "randaug_magnitude": cfg["randaug_magnitude"],
            "cutmix_alpha": cfg["cutmix_alpha"],
            "mixup_alpha": cfg["mixup_alpha"],
            "cutmix_mixup_prob": cfg["cutmix_mixup_prob"],
            "switch_prob": cfg["switch_prob"],
        },
        "metriche": {},
        "epoche_log": [],
        "tempo_totale_sec": None,
    }
    if nova_backend:
        log_data["configurazione"]["nova_backend"] = nova_backend

    # --- Training loop ---
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            cutmix_mixup=cutmix_mixup)
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "lr": round(current_lr, 8),
            "epoch_time_sec": round(epoch_time, 1),
        }
        if activation == "nova":
            for m in model.modules():
                if isinstance(m, (NOVACuda, NOVAPython)):
                    epoch_log["beta"] = round(m.beta.item(), 6)
                    break

        log_data["epoche_log"].append(epoch_log)

        # Salvataggio incrementale
        log_data["metriche"] = {
            "best_val_acc": round(best_val_acc, 2),
            "final_val_acc": round(val_acc, 2),
            "final_train_loss": round(train_loss, 4),
            "final_val_loss": round(val_loss, 4),
            "final_train_acc": round(train_acc, 2),
        }
        log_data["tempo_totale_sec"] = round(time.time() - start_time, 2)
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        print(f"[{scale.upper()}/{activation.upper()}/CNX] "
              f"Epoch {epoch+1:03d}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  "
              f"lr={current_lr:.6f}  ({epoch_time:.1f}s)"
              + (f"  β={epoch_log.get('beta', '')}" if activation == "nova" else ""))

    elapsed = time.time() - start_time

    log_data["tempo_totale_sec"] = round(elapsed, 2)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    print(f"\n{'='*60}")
    print(f"  RISULTATI — {scale.upper()} / {activation.upper()} / ConvNeXt")
    print(f"  Parametri: {num_params:,}")
    print(f"  Best val acc: {best_val_acc:.2f}%")
    print(f"  Tempo totale: {elapsed:.1f}s")
    print(f"  Log: {log_path}")
    print(f"{'='*60}\n")

    del model, optimizer, scaler, scheduler, criterion
    del train_loader, test_loader
    torch.cuda.empty_cache()
    print(f"[{scale.upper()}/{activation.upper()}/CNX] VRAM liberata.")


# ==============================================================
# PLOT PER IL PAPER
# ==============================================================

def generate_plots():
    """Genera plot ConvNeXt e confronto ViT vs ConvNeXt."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results_dir = _get_results_dir()
    scales = ["tiny", "small", "base"]
    activations = ["nova", "gelu"]

    def load_latest(pattern):
        files = sorted(glob.glob(os.path.join(results_dir, pattern)))
        if not files:
            return None
        with open(files[-1]) as f:
            return json.load(f)

    cnx_data = {}
    for scale in scales:
        for act in activations:
            d = load_latest(f"convnext_cifar100_{scale}_{act}_*.json")
            if d:
                cnx_data[(scale, act)] = d

    if not cnx_data:
        print("[PLOT] Nessun risultato ConvNeXt trovato.")
        return

    colors = {"nova": "#E63946", "gelu": "#457B9D"}
    scale_labels_cnx = {}
    for scale in scales:
        key = (scale, "nova") if (scale, "nova") in cnx_data else (scale, "gelu")
        if key in cnx_data:
            nparams = cnx_data[key]["configurazione"]["num_params"]
            scale_labels_cnx[scale] = f"{scale.capitalize()} ({nparams/1e6:.1f}M)"
        else:
            scale_labels_cnx[scale] = scale.capitalize()

    # --- PLOT 1: Training curves per scala ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("ConvNeXt su CIFAR-100 — Training Curves (100 epoche)",
                 fontsize=14, fontweight='bold')

    for col, scale in enumerate(scales):
        ax_acc = axes[0, col]
        ax_loss = axes[1, col]

        for act in activations:
            key = (scale, act)
            if key not in cnx_data:
                continue
            data = cnx_data[key]
            eps = [e["epoch"] for e in data["epoche_log"]]
            val_accs = [e["val_acc"] for e in data["epoche_log"]]
            train_losses = [e["train_loss"] for e in data["epoche_log"]]
            val_losses = [e["val_loss"] for e in data["epoche_log"]]
            best = data["metriche"]["best_val_acc"]

            ax_acc.plot(eps, val_accs, color=colors[act], linewidth=1.5,
                        label=f"{act.upper()} (best {best}%)")
            ax_loss.plot(eps, train_losses, color=colors[act],
                         linewidth=1.2, linestyle='--', alpha=0.7,
                         label=f"{act.upper()} train")
            ax_loss.plot(eps, val_losses, color=colors[act],
                         linewidth=1.5, label=f"{act.upper()} val")

        ax_acc.set_title(scale_labels_cnx.get(scale, scale), fontsize=12)
        ax_acc.set_ylabel("Val Accuracy (%)" if col == 0 else "")
        ax_acc.legend(fontsize=8)
        ax_acc.grid(True, alpha=0.3)

        ax_loss.set_xlabel("Epoca")
        ax_loss.set_ylabel("Loss" if col == 0 else "")
        ax_loss.legend(fontsize=8)
        ax_loss.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "plot_convnext_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # --- PLOT 2: Scaling curve ConvNeXt ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("ConvNeXt CIFAR-100: Scaling Curve (NOVA vs GELU)",
                 fontsize=13, fontweight='bold')

    for act in activations:
        params_list, acc_list = [], []
        for scale in scales:
            key = (scale, act)
            if key not in cnx_data:
                continue
            d = cnx_data[key]
            params_list.append(d["configurazione"]["num_params"] / 1e6)
            acc_list.append(d["metriche"]["best_val_acc"])
        if params_list:
            ax.plot(params_list, acc_list, color=colors[act],
                    marker='s', markersize=8, linewidth=2,
                    label=f"{act.upper()}")

    ax.set_xlabel("Parametri (M)", fontsize=11)
    ax.set_ylabel("Best Val Accuracy (%)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "plot_convnext_scaling_curve.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # --- PLOT 3: ViT vs ConvNeXt — NOVA advantage ---
    vit_data = {}
    for scale in scales:
        for act in activations:
            # Prova v3, poi v2
            d = load_latest(f"vit_scaling_v3_{scale}_{act}_*.json")
            if d is None:
                d = load_latest(f"vit_scaling_v2_{scale}_{act}_*.json")
            if d:
                vit_data[(scale, act)] = d

    if vit_data:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.set_title("NOVA advantage (pp): ViT vs ConvNeXt su CIFAR-100",
                      fontsize=13, fontweight='bold')

        x = np.arange(len(scales))
        width = 0.35

        for i, (arch_name, data_dict) in enumerate([
            ("ViT", vit_data),
            ("ConvNeXt", cnx_data),
        ]):
            deltas = []
            for scale in scales:
                nova_key = (scale, "nova")
                gelu_key = (scale, "gelu")
                if nova_key in data_dict and gelu_key in data_dict:
                    nova_acc = data_dict[nova_key]["metriche"]["best_val_acc"]
                    gelu_acc = data_dict[gelu_key]["metriche"]["best_val_acc"]
                    deltas.append(nova_acc - gelu_acc)
                else:
                    deltas.append(0)

            color = "#E63946" if i == 0 else "#264653"
            bars = ax.bar(x + (i - 0.5) * width, deltas, width,
                          label=arch_name, color=color,
                          edgecolor='black', linewidth=0.5)
            for bar, delta in zip(bars, deltas):
                if delta != 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.15,
                            f'+{delta:.1f}', ha='center', va='bottom',
                            fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in scales])
        ax.set_ylabel("NOVA - GELU (pp)", fontsize=11)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        path = os.path.join(results_dir, "plot_vit_vs_convnext_nova_advantage.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[PLOT] Salvato: {path}")

    # --- PLOT 4: Beta evolution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Evoluzione β (NOVA) — ConvNeXt CIFAR-100",
                 fontsize=13, fontweight='bold')

    scale_colors = {"tiny": "#264653", "small": "#2A9D8F", "base": "#E76F51"}
    for scale in scales:
        key = (scale, "nova")
        if key not in cnx_data:
            continue
        data = cnx_data[key]
        eps = [e["epoch"] for e in data["epoche_log"]]
        betas = [e.get("beta", None) for e in data["epoche_log"]]
        if betas[0] is not None:
            nparams = data["configurazione"]["num_params"]
            ax.plot(eps, betas, color=scale_colors[scale], linewidth=2,
                    label=f"{scale.capitalize()} ({nparams/1e6:.1f}M)")

    ax.set_xlabel("Epoca", fontsize=11)
    ax.set_ylabel("β", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "plot_convnext_beta_evolution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    print(f"\n[PLOT] Tutti i plot generati in: {results_dir}")


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
    print("[LAUNCHER] Pre-download CIFAR-100...")
    data_root = _get_data_root()
    torchvision.datasets.CIFAR100(root=data_root, train=True, download=True)
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
        print(f"  SCALA: {scale.upper()} "
              f"(dims={cfg['dims']}, depths={cfg['depths']})")
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
                       "--scale", scale,
                       "--activation", act,
                       "--gpu", str(gpu)]
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

    print("\n[LAUNCHER] Generazione plot...")
    generate_plots()

    print(f"\n{'='*60}")
    print("  CONVNEXT CIFAR-100 SCALING STUDY COMPLETATO")
    print(f"{'='*60}")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ConvNeXt Scaling su CIFAR-100: NOVA vs GELU")
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
