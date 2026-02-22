#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESPERIMENTO: Mini-ViT su CIFAR-100
===================================
NOVA vs GELU — Mixed Precision (FP16/FP32) su 2× NVIDIA T4

Uso:
    # Lancia entrambi gli esperimenti in parallelo su 2 GPU:
    python vit_cifar100.py

    # Lancia un singolo esperimento:
    python vit_cifar100.py --activation nova --gpu 0
    python vit_cifar100.py --activation gelu --gpu 1
"""

import argparse
import subprocess
import sys
import os
import time
import json
import math
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as transforms

# ==============================================================
# CONFIGURAZIONE ESPERIMENTO
# ==============================================================
SEED = 42
DATASET = "CIFAR-100"
HARDWARE = "NVIDIA T4"
EPOCHS = 100
BATCH_SIZE = 1024
LR = 3e-3
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 10
NUM_WORKERS = 4

# Architettura Mini-ViT
PATCH_SIZE = 4
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
MLP_RATIO = 4
DROPOUT = 0.1
NUM_CLASSES = 100
IMG_SIZE = 32

# ==============================================================
# RIPRODUCIBILITA'
# ==============================================================
def set_seed(seed: int) -> None:
    """Fissa tutti i seed per riproducibilità."""
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

# --- Kernel CUDA (compilazione JIT) ---
_nova_cuda_ext = None

def _compile_nova_cuda():
    """Compila il kernel CUDA fuso per NOVA. Ritorna None se fallisce."""
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

    nova_ext = load_inline(
        name='nova_cuda_ext',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['nova_cuda_forward', 'nova_cuda_backward'],
        with_cuda=True,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
    )
    return nova_ext


class _NOVAFunction(torch.autograd.Function):
    """Autograd wrapper per il kernel CUDA fuso."""
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
    """NOVA con kernel CUDA fuso. β apprendibile."""
    def __init__(self, nova_ext, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.nova_ext = nova_ext

    def forward(self, x):
        return _NOVAFunction.apply(x, self.beta, self.nova_ext)


class NOVAPython(nn.Module):
    """NOVA in Python puro (fallback). β apprendibile, gradiente via autograd."""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def forward(self, x):
        bx = self.beta * x
        return x * torch.sigmoid(bx) - x / (1.0 + bx ** 2)


def make_nova(beta=1.0):
    """Crea un'istanza NOVA: preferisce CUDA, fallback Python."""
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
# MODELLO: Mini Vision Transformer
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
        # (B, C, H, W) -> (B, num_patches, embed_dim)
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
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout, act_layer):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
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
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes,
                 embed_dim, num_heads, num_layers, mlp_ratio, dropout,
                 act_layer):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, act_layer)
            for _ in range(num_layers)
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
# DATI: CIFAR-100
# ==============================================================

def get_cifar100_loaders(batch_size, num_workers):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_root = os.path.join(os.path.dirname(__file__), "..", "data")
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
# TRAINING & VALUTAZIONE
# ==============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
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


# ==============================================================
# LR SCHEDULER: Linear Warmup + Cosine Decay
# ==============================================================

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

def run_experiment(activation: str, gpu: int) -> None:
    set_seed(SEED)

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    experiment_name = f"vit_cifar100_{activation}"
    model_desc = (f"Mini-ViT ({NUM_LAYERS}L, {EMBED_DIM}d, {NUM_HEADS}h, "
                  f"MLP×{MLP_RATIO}) + {activation.upper()}")

    print(f"\n{'='*60}")
    print(f"  ESPERIMENTO: {experiment_name}")
    print(f"  GPU: {gpu} ({torch.cuda.get_device_name(device)})")
    print(f"  Attivazione: {activation.upper()}")
    print(f"  Epoche: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")
    print(f"  Mixed Precision: FP16 (calcoli critici in FP32)")
    print(f"  Seed: {SEED}")
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
    model = MiniViT(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=3,
        num_classes=NUM_CLASSES, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, mlp_ratio=MLP_RATIO, dropout=DROPOUT,
        act_layer=act_layer,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parametri totali: {num_params:,}")

    # --- Dati ---
    train_loader, test_loader = get_cifar100_loaders(BATCH_SIZE, NUM_WORKERS)

    # --- Ottimizzatore, Loss, Scheduler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
    scaler = GradScaler()

    # --- Log setup ---
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(results_dir, f"{experiment_name}_{timestamp}.json")

    log_data = {
        "esperimento": experiment_name,
        "data": datetime.now().strftime("%Y-%m-%d"),
        "hardware": f"{HARDWARE} (GPU {gpu})",
        "seed": SEED,
        "dataset": DATASET,
        "modello": model_desc,
        "obiettivo": f"Test accuracy a {EPOCHS} epoche, confronto NOVA vs GELU",
        "configurazione": {
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
            "label_smoothing": 0.1,
            "mixed_precision": "fp16",
            "num_params": num_params,
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

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

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
        # Logga beta per NOVA
        if activation == "nova":
            for m in model.modules():
                if isinstance(m, (NOVACuda, NOVAPython)):
                    epoch_log["beta"] = round(m.beta.item(), 6)
                    break

        log_data["epoche_log"].append(epoch_log)

        print(f"[Epoch {epoch+1:02d}/{EPOCHS}] "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  "
              f"lr={current_lr:.6f}  ({epoch_time:.1f}s)"
              + (f"  β={epoch_log.get('beta', '')}" if activation == "nova" else ""))

    elapsed = time.time() - start_time

    # --- Metriche finali ---
    log_data["metriche"] = {
        "best_val_acc": round(best_val_acc, 2),
        "final_val_acc": round(val_acc, 2),
        "final_train_loss": round(train_loss, 4),
        "final_val_loss": round(val_loss, 4),
        "final_train_acc": round(train_acc, 2),
    }
    log_data["tempo_totale_sec"] = round(elapsed, 2)

    # --- Salvataggio ---
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  RISULTATI — {activation.upper()}")
    print(f"  Best val acc: {best_val_acc:.2f}%")
    print(f"  Final val acc: {val_acc:.2f}%")
    print(f"  Tempo totale: {elapsed:.1f}s")
    print(f"  Log: {log_path}")
    print(f"{'='*60}\n")


# ==============================================================
# LAUNCHER: parallelo su 2 GPU
# ==============================================================

ALL_ACTIVATIONS = ["nova", "gelu", "silu", "mish", "relu"]


def launch_all():
    """Lancia tutte le attivazioni in parallelo (2 alla volta sulle 2 GPU)."""
    global _nova_cuda_ext

    # 1. Pre-download dataset (evita race condition tra sottoprocessi)
    print("[LAUNCHER] Pre-download dataset CIFAR-100...")
    data_root = os.path.join(os.path.dirname(__file__), "..", "data")
    torchvision.datasets.CIFAR100(root=data_root, train=True, download=True)
    torchvision.datasets.CIFAR100(root=data_root, train=False, download=True)
    print("[LAUNCHER] Dataset pronto.")

    # 2. Pre-compilazione kernel CUDA (cachato per i sottoprocessi)
    print("[LAUNCHER] Pre-compilazione kernel CUDA NOVA (30-60s)...")
    try:
        _nova_cuda_ext = _compile_nova_cuda()
        print("[LAUNCHER] Kernel CUDA compilato e cachato.")
    except Exception as e:
        print(f"[LAUNCHER] ATTENZIONE: Compilazione CUDA fallita: {e}")
        print("[LAUNCHER] NOVA userà il fallback Python puro.")

    # 3. Lancia a coppie sulle 2 GPU
    script = os.path.abspath(__file__)
    pairs = [(ALL_ACTIVATIONS[i], ALL_ACTIVATIONS[i + 1] if i + 1 < len(ALL_ACTIVATIONS) else None)
             for i in range(0, len(ALL_ACTIVATIONS), 2)]

    for pair in pairs:
        procs = []
        for gpu, act in enumerate(pair):
            if act is None:
                continue
            cmd = [sys.executable, script, "--activation", act, "--gpu", str(gpu)]
            print(f"\n[LAUNCHER] Avvio: {' '.join(cmd)}")
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
            procs.append((act, p))

        for act, p in procs:
            p.wait()
            if p.returncode != 0:
                print(f"[LAUNCHER] ERRORE: {act} terminato con codice {p.returncode}")
            else:
                print(f"[LAUNCHER] {act.upper()} completato con successo.")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-ViT CIFAR-100: NOVA vs GELU/SiLU/Mish/ReLU")
    parser.add_argument("--activation", type=str,
                        choices=["nova", "gelu", "silu", "mish", "relu"],
                        help="Funzione di attivazione (ometti per lanciare tutti)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="ID della GPU (default: 0)")
    args = parser.parse_args()

    if args.activation is None:
        # Modalità parallela: lancia entrambi
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            print(f"[ATTENZIONE] Solo {n_gpus} GPU disponibili. "
                  "Lancia manualmente con --activation e --gpu.")
            sys.exit(1)
        launch_all()
    else:
        run_experiment(args.activation, args.gpu)
