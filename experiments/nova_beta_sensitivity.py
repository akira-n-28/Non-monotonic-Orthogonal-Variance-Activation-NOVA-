#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESPERIMENTO: Sensibilità a β₀ (inizializzazione di NOVA)
==========================================================
Studio della robustezza di NOVA rispetto al valore iniziale del parametro
apprendibile β. Usa ViT-Small v2 su CIFAR-100 (configurazione con il
vantaggio NOVA più chiaro: +8.35 pp, β finale ≈ 0.45).

Domanda: β converge allo stesso valore indipendentemente da β₀?

β₀ ∈ {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0}

Uso:
    # Lancia tutti i β₀ (a coppie su 2 GPU):
    python nova_beta_sensitivity.py

    # Singolo β₀:
    python nova_beta_sensitivity.py --beta-init 0.5 --gpu 0

    # Solo plot dai log esistenti:
    python nova_beta_sensitivity.py --plot-only
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

# Architettura fissa: ViT-Small v2 (la configurazione con vantaggio NOVA massimo)
EMBED_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 100
WARMUP_EPOCHS = 15
DROPOUT = 0.1
WEIGHT_DECAY = 0.05
DROP_PATH_RATE = 0.2

# Regolarizzazione DeiT-style (identica a vit_scaling_v2.py Small)
CUTMIX_ALPHA = 1.0
MIXUP_ALPHA = 0.8
CUTMIX_MIXUP_PROB = 1.0
SWITCH_PROB = 0.5
RANDAUG_NUM_OPS = 2
RANDAUG_MAGNITUDE = 9

# Valori di β₀ da testare
BETA_INITS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


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
        name='nova_cuda_ext_beta',
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

def get_cifar100_loaders(batch_size, num_workers):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=RANDAUG_NUM_OPS,
                               magnitude=RANDAUG_MAGNITUDE),
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

def run_experiment(beta_init: float, gpu: int) -> None:
    set_seed(SEED)

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    # Formatta β₀ per il nome file (0.1 -> "0p10", 1.0 -> "1p00")
    beta_str = f"{beta_init:.2f}".replace(".", "p")
    experiment_name = f"nova_beta_sensitivity_b{beta_str}"
    model_desc = (f"ViT-Small ({NUM_LAYERS}L, {EMBED_DIM}d, {NUM_HEADS}h, "
                  f"MLP×{MLP_RATIO}) + NOVA(β₀={beta_init})")

    print(f"\n{'='*60}")
    print(f"  ESPERIMENTO: {experiment_name}")
    print(f"  GPU: {gpu} ({torch.cuda.get_device_name(device)})")
    print(f"  β₀ = {beta_init}")
    print(f"  Architettura: ViT-Small v2 ({NUM_LAYERS}L, {EMBED_DIM}d, {NUM_HEADS}h)")
    print(f"  Epoche: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")
    print(f"  Regolarizzazione: DeiT-style (RandAug, CutMix+Mixup, DropPath)")
    print(f"  Mixed Precision: FP16")
    print(f"  Seed: {SEED}")
    print(f"{'='*60}\n")

    # --- Activation: NOVA con β₀ specificato ---
    act_layer, nova_backend = make_nova(beta=beta_init)
    print(f"[INFO] NOVA backend: {nova_backend}, β₀ = {beta_init}")

    # --- Modello ---
    model = ScalableViT(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=3,
        num_classes=NUM_CLASSES, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, mlp_ratio=MLP_RATIO, dropout=DROPOUT,
        act_layer=act_layer, drop_path_rate=DROP_PATH_RATE,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parametri totali: {num_params:,}")

    # --- Dati ---
    train_loader, test_loader = get_cifar100_loaders(BATCH_SIZE, NUM_WORKERS)

    # --- CutMix/Mixup ---
    cutmix_mixup = CutMixMixupCollator(
        num_classes=NUM_CLASSES,
        cutmix_alpha=CUTMIX_ALPHA,
        mixup_alpha=MIXUP_ALPHA,
        prob=CUTMIX_MIXUP_PROB,
        switch_prob=SWITCH_PROB,
    )

    # --- Ottimizzatore, Loss, Scheduler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
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
        "obiettivo": f"Sensibilità a β₀: NOVA con β₀={beta_init} su ViT-Small v2",
        "configurazione": {
            "beta_init": beta_init,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_epochs": WARMUP_EPOCHS,
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "mlp_ratio": MLP_RATIO,
            "dropout": DROPOUT,
            "drop_path_rate": DROP_PATH_RATE,
            "label_smoothing": LABEL_SMOOTHING,
            "mixed_precision": "fp16",
            "num_params": num_params,
            "nova_backend": nova_backend,
            "randaug_num_ops": RANDAUG_NUM_OPS,
            "randaug_magnitude": RANDAUG_MAGNITUDE,
            "cutmix_alpha": CUTMIX_ALPHA,
            "mixup_alpha": MIXUP_ALPHA,
        },
        "metriche": {},
        "epoche_log": [],
        "tempo_totale_sec": None,
    }

    # --- Training loop ---
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(EPOCHS):
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

        # Raccogli β da tutti i layer NOVA (tutti dovrebbero convergere allo
        # stesso valore dato che condividono lo stesso modulo act_layer)
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

        print(f"[β₀={beta_init}] "
              f"Epoch {epoch+1:03d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_acc={val_acc:.2f}%  "
              f"β={beta_val}  lr={current_lr:.6f}  ({epoch_time:.1f}s)")

    elapsed = time.time() - start_time

    # --- Salvataggio finale ---
    log_data["tempo_totale_sec"] = round(elapsed, 2)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    print(f"\n{'='*60}")
    print(f"  RISULTATI — β₀ = {beta_init}")
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

def generate_plots():
    """Genera 3 plot per l'analisi di sensibilità a β₀."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
    except ImportError:
        print("[PLOT] matplotlib non disponibile, skip plot.")
        return

    results_dir = _get_results_dir()

    # --- Carica log ---
    logs = {}  # beta_init -> log_data
    for f in sorted(os.listdir(results_dir)):
        if f.startswith("nova_beta_sensitivity_b") and f.endswith(".json"):
            path = os.path.join(results_dir, f)
            with open(path) as fh:
                data = json.load(fh)
            beta_init = data["configurazione"]["beta_init"]
            # Tieni il log più recente per ogni β₀
            logs[beta_init] = data

    if not logs:
        print("[PLOT] Nessun log di sensibilità β trovato.")
        return

    beta_inits = sorted(logs.keys())
    print(f"[PLOT] Trovati {len(beta_inits)} valori di β₀: {beta_inits}")

    # Colormap: dal blu (β₀ basso) al rosso (β₀ alto)
    cmap = get_cmap("coolwarm")
    beta_min, beta_max = min(beta_inits), max(beta_inits)

    def beta_color(b):
        if beta_max == beta_min:
            return cmap(0.5)
        return cmap((b - beta_min) / (beta_max - beta_min))

    # ===========================================================
    # PLOT 1: Convergenza di β — β(t) vs epoca per ogni β₀
    # ===========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Convergenza di β per diversi β₀ — ViT-Small v2 CIFAR-100",
                 fontsize=13, fontweight='bold')

    for b0 in beta_inits:
        data = logs[b0]
        epochs = [e["epoch"] for e in data["epoche_log"]]
        betas = [e["beta"] for e in data["epoche_log"]]
        ax.plot(epochs, betas, color=beta_color(b0), linewidth=2.0,
                label=f"β₀={b0}")

    ax.set_xlabel("Epoca", fontsize=11)
    ax.set_ylabel("β", fontsize=11)
    ax.legend(fontsize=9, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "plot_beta_sensitivity_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # ===========================================================
    # PLOT 2: Best val accuracy vs β₀ (bar chart + linea)
    # ===========================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Best Val Accuracy vs β₀ — ViT-Small v2 CIFAR-100",
                 fontsize=13, fontweight='bold')

    accs = [logs[b0]["metriche"]["best_val_acc"] for b0 in beta_inits]
    bar_colors = [beta_color(b0) for b0 in beta_inits]
    x_pos = np.arange(len(beta_inits))

    bars = ax.bar(x_pos, accs, color=bar_colors, edgecolor='black',
                  linewidth=0.5, width=0.6)

    # Annota ogni barra con il valore
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold')

    # Range e best evidenziato
    acc_min, acc_max = min(accs), max(accs)
    acc_range = acc_max - acc_min
    ax.axhline(y=acc_max, color='green', linestyle='--', alpha=0.5,
               label=f'Max: {acc_max:.2f}%')
    ax.axhline(y=acc_min, color='red', linestyle='--', alpha=0.5,
               label=f'Min: {acc_min:.2f}%')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{b0}' for b0 in beta_inits])
    ax.set_xlabel("β₀ (inizializzazione)", fontsize=11)
    ax.set_ylabel("Best Val Accuracy (%)", fontsize=11)
    # Y-axis: zoom per evidenziare le differenze
    y_pad = max(acc_range * 0.5, 1.0)
    ax.set_ylim(acc_min - y_pad, acc_max + y_pad)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(results_dir, "plot_beta_sensitivity_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # ===========================================================
    # PLOT 3: Training dynamics — val loss vs epoca per ogni β₀
    # ===========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Val Loss durante il training per diversi β₀",
                 fontsize=13, fontweight='bold')

    for b0 in beta_inits:
        data = logs[b0]
        epochs = [e["epoch"] for e in data["epoche_log"]]
        val_losses = [e["val_loss"] for e in data["epoche_log"]]
        ax.plot(epochs, val_losses, color=beta_color(b0), linewidth=1.8,
                label=f"β₀={b0}")

    ax.set_xlabel("Epoca", fontsize=11)
    ax.set_ylabel("Val Loss", fontsize=11)
    ax.legend(fontsize=9, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "plot_beta_sensitivity_val_loss.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Salvato: {path}")

    # ===========================================================
    # Tabella riassuntiva (stampa a terminale)
    # ===========================================================
    print(f"\n{'='*70}")
    print(f"  RIEPILOGO SENSIBILITA' β₀")
    print(f"{'='*70}")
    print(f"  {'β₀':>6}  {'Best Acc':>10}  {'β finale':>10}  {'Δ vs β₀=1.0':>12}")
    print(f"  {'-'*44}")

    ref_acc = logs.get(1.0, {}).get("metriche", {}).get("best_val_acc", None)
    for b0 in beta_inits:
        m = logs[b0]["metriche"]
        acc = m["best_val_acc"]
        beta_f = m.get("final_beta", "?")
        delta = f"{acc - ref_acc:+.2f}" if ref_acc is not None else "N/A"
        print(f"  {b0:>6.2f}  {acc:>9.2f}%  {beta_f:>10}  {delta:>12}")

    print(f"  {'-'*44}")
    print(f"  Range: {acc_max - acc_min:.2f} pp (min {acc_min:.2f}%, max {acc_max:.2f}%)")
    print(f"{'='*70}\n")


# ==============================================================
# LAUNCHER: parallelo su 2 GPU
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
        print("[LAUNCHER] NOVA userà il fallback Python puro.")

    # 3. Lancia a coppie sulle 2 GPU
    script = os.path.abspath(__file__)

    pairs = []
    for i in range(0, len(BETA_INITS), 2):
        pair = [BETA_INITS[i]]
        if i + 1 < len(BETA_INITS):
            pair.append(BETA_INITS[i + 1])
        pairs.append(pair)

    for pair in pairs:
        procs = []
        for gpu, beta_init in enumerate(pair):
            cmd = [sys.executable, script,
                   "--beta-init", str(beta_init), "--gpu", str(gpu)]
            print(f"\n[LAUNCHER] Avvio: {' '.join(cmd)}")
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
            procs.append((beta_init, p))

        for beta_init, p in procs:
            p.wait()
            if p.returncode != 0:
                print(f"[LAUNCHER] ERRORE: β₀={beta_init} terminato con codice {p.returncode}")
            else:
                print(f"[LAUNCHER] β₀={beta_init} completato con successo.")

    # 4. Genera plot
    print("\n[LAUNCHER] Generazione plot...")
    generate_plots()
    print("[LAUNCHER] Esperimento di sensibilità β₀ completato.")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NOVA β₀ sensitivity: ViT-Small v2 CIFAR-100")
    parser.add_argument("--beta-init", type=float,
                        help="Valore di β₀ (ometti per lanciare tutti)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="ID della GPU (default: 0)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Solo generazione plot dai log esistenti")
    args = parser.parse_args()

    if args.plot_only:
        generate_plots()
    elif args.beta_init is not None:
        run_experiment(args.beta_init, args.gpu)
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            print(f"[ATTENZIONE] Solo {n_gpus} GPU disponibili.")
            print("Lancia manualmente con --beta-init e --gpu.")
            sys.exit(1)
        launch_all()
