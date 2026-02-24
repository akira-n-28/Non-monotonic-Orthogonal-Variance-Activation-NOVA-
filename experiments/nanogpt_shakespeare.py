#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESPERIMENTO: Nano-GPT su TinyShakespeare
==========================================
NOVA vs GELU/SiLU/Mish/ReLU — Scaling Study su Language Modeling
Decoder-only Transformer con tokenizzazione char-level.

Uso:
    # Lancia tutti gli esperimenti (tutte le scale e attivazioni) su 2 GPU:
    python nanogpt_shakespeare.py

    # Lancia un singolo esperimento:
    python nanogpt_shakespeare.py --scale small --activation nova --gpu 0

    # Solo plot dai log esistenti:
    python nanogpt_shakespeare.py --plot-only
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
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

# ==============================================================
# CONFIGURAZIONE ESPERIMENTO
# ==============================================================
SEED = 42
DATASET = "TinyShakespeare"
HARDWARE = "NVIDIA T4"
MAX_ITERS = 5000
EVAL_INTERVAL = 250
EVAL_ITERS = 200
WARMUP_ITERS = 500
NUM_WORKERS = 0
GRAD_CLIP = 1.0

# Context e tokenizzazione
BLOCK_SIZE = 256     # context length
BATCH_SIZE = 64      # batch size

# Scaling configs
SCALING_CONFIGS = {
    "tiny": {
        "n_layer": 4, "n_embd": 256, "n_head": 4,
        "dropout": 0.1, "lr": 1e-3, "weight_decay": 0.01,
    },
    "small": {
        "n_layer": 6, "n_embd": 384, "n_head": 6,
        "dropout": 0.1, "lr": 6e-4, "weight_decay": 0.01,
    },
    "base": {
        "n_layer": 8, "n_embd": 512, "n_head": 8,
        "dropout": 0.1, "lr": 3e-4, "weight_decay": 0.01,
    },
}

ALL_ACTIVATIONS = ["nova", "gelu", "silu", "mish", "relu"]
ALL_SCALES = ["tiny", "small", "base"]


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
# DATASET: TinyShakespeare (char-level)
# ==============================================================

def download_tinyshakespeare(data_root):
    """Scarica TinyShakespeare se non presente."""
    os.makedirs(data_root, exist_ok=True)
    filepath = os.path.join(data_root, "tinyshakespeare.txt")
    if not os.path.exists(filepath):
        print("[DATA] Download TinyShakespeare...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, filepath)
        print(f"[DATA] Salvato in {filepath}")
    return filepath


def load_dataset(data_root):
    """Carica e tokenizza TinyShakespeare (char-level)."""
    filepath = download_tinyshakespeare(data_root)
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    # Split 90/10
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_size, stoi, itos


def get_batch(data, block_size, batch_size, device):
    """Genera un batch random di sequenze."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# ==============================================================
# MODELLO: Nano-GPT (Decoder-only Transformer)
# ==============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim ** -0.5

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer("bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class GPTBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size, act_layer):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            act_layer,
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_embd, n_head,
                 dropout, act_layer):
        super().__init__()
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            GPTBlock(n_embd, n_head, dropout, block_size, act_layer)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.token_emb.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ==============================================================
# LR SCHEDULER: Linear Warmup + Cosine Decay
# ==============================================================

def get_lr(it, warmup_iters, max_iters, max_lr, min_lr=1e-5):
    """Cosine decay with linear warmup."""
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it >= max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ==============================================================
# VALUTAZIONE
# ==============================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size,
                  device, eval_iters):
    """Stima train/val loss su eval_iters batch random."""
    model.eval()
    out = {}
    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            with autocast(device_type='cuda', dtype=torch.float16):
                _, loss = model(x, y)
            losses.append(loss.item())
        out[split_name] = np.mean(losses)
    model.train()
    return out


# ==============================================================
# GENERAZIONE CAMPIONI
# ==============================================================

@torch.no_grad()
def generate_sample(model, itos, device, max_tokens=500, temperature=0.8):
    """Genera un campione di testo dal modello."""
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=max_tokens,
                               temperature=temperature, top_k=40)
    text = ''.join([itos[i] for i in generated[0].tolist()])
    model.train()
    return text


# ==============================================================
# ESPERIMENTO PRINCIPALE
# ==============================================================

def run_experiment(scale: str, activation: str, gpu: int) -> None:
    set_seed(SEED)

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    config = SCALING_CONFIGS[scale]
    n_layer = config["n_layer"]
    n_embd = config["n_embd"]
    n_head = config["n_head"]
    dropout = config["dropout"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]

    experiment_name = f"nanogpt_{scale}_{activation}"
    model_desc = (f"NanoGPT-{scale.capitalize()} ({n_layer}L, {n_embd}d, "
                  f"{n_head}h, MLP×4) + {activation.upper()}")

    print(f"\n{'='*60}")
    print(f"  ESPERIMENTO: {experiment_name}")
    print(f"  GPU: {gpu} ({torch.cuda.get_device_name(device)})")
    print(f"  Scala: {scale.upper()}")
    print(f"  Attivazione: {activation.upper()}")
    print(f"  Iterazioni: {MAX_ITERS}, Batch: {BATCH_SIZE}, LR: {lr}")
    print(f"  Context: {BLOCK_SIZE}, Mixed Precision: FP16")
    print(f"  Seed: {SEED}")
    print(f"{'='*60}\n")

    # --- Dataset ---
    data_root = _get_data_root()
    train_data, val_data, vocab_size, stoi, itos = load_dataset(data_root)
    print(f"[DATA] Vocab size: {vocab_size}, Train: {len(train_data):,} chars, "
          f"Val: {len(val_data):,} chars")

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
    model = NanoGPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        dropout=dropout,
        act_layer=act_layer,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    # Weight tying: non contare lm_head separatamente
    num_params_no_emb = sum(p.numel() for n, p in model.named_parameters()
                            if 'token_emb' not in n and 'pos_emb' not in n)
    print(f"[INFO] Parametri totali: {num_params:,} (non-embedding: {num_params_no_emb:,})")

    # --- Ottimizzatore ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.95))
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
        "obiettivo": f"Val loss a {MAX_ITERS} iterazioni, confronto multi-attivazione",
        "configurazione": {
            "max_iters": MAX_ITERS,
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_iters": WARMUP_ITERS,
            "grad_clip": GRAD_CLIP,
            "n_layer": n_layer,
            "n_embd": n_embd,
            "n_head": n_head,
            "dropout": dropout,
            "vocab_size": vocab_size,
            "mixed_precision": "fp16",
            "num_params": num_params,
            "num_params_no_emb": num_params_no_emb,
            "scale": scale,
        },
        "metriche": {},
        "iter_log": [],
        "campioni_testo": [],
        "tempo_totale_sec": None,
    }
    if nova_backend:
        log_data["configurazione"]["nova_backend"] = nova_backend

    # --- Training loop ---
    best_val_loss = float('inf')
    start_time = time.time()
    model.train()

    for it in range(MAX_ITERS):
        # LR scheduling
        current_lr = get_lr(it, WARMUP_ITERS, MAX_ITERS, lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Get batch
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)

        # Forward + backward
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=torch.float16):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # --- Evaluation ---
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data, BLOCK_SIZE,
                                   BATCH_SIZE, device, EVAL_ITERS)
            train_loss = losses["train"]
            val_loss = losses["val"]
            perplexity = math.exp(val_loss) if val_loss < 20 else float('inf')

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            iter_log = {
                "iter": it,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "perplexity": round(perplexity, 4),
                "lr": round(current_lr, 8),
                "elapsed_sec": round(time.time() - start_time, 1),
            }

            # Beta tracking per NOVA
            if activation == "nova":
                for m in model.modules():
                    if isinstance(m, (NOVACuda, NOVAPython)):
                        iter_log["beta"] = round(m.beta.item(), 6)
                        break

            log_data["iter_log"].append(iter_log)

            # Genera campione di testo ogni 1000 iterazioni
            if it % 1000 == 0 or it == MAX_ITERS - 1:
                sample = generate_sample(model, itos, device,
                                         max_tokens=200, temperature=0.8)
                log_data["campioni_testo"].append({
                    "iter": it,
                    "testo": sample[:500],  # tronca per sicurezza
                })

            # Aggiorna metriche
            log_data["metriche"] = {
                "best_val_loss": round(best_val_loss, 6),
                "best_perplexity": round(math.exp(best_val_loss), 4) if best_val_loss < 20 else None,
                "final_val_loss": round(val_loss, 6),
                "final_train_loss": round(train_loss, 6),
                "final_perplexity": round(perplexity, 4),
            }
            log_data["tempo_totale_sec"] = round(time.time() - start_time, 2)

            # Salvataggio incrementale
            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            beta_str = f"  beta={iter_log.get('beta', '')}" if activation == "nova" else ""
            print(f"[Iter {it:5d}/{MAX_ITERS}] "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"ppl={perplexity:.2f}  lr={current_lr:.6f}"
                  f"{beta_str}  ({iter_log['elapsed_sec']:.0f}s)")

    elapsed = time.time() - start_time

    # --- Salvataggio finale ---
    log_data["tempo_totale_sec"] = round(elapsed, 2)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    best_ppl = math.exp(best_val_loss) if best_val_loss < 20 else float('inf')
    print(f"\n{'='*60}")
    print(f"  RISULTATI — {scale.upper()} / {activation.upper()} / NanoGPT")
    print(f"  Parametri: {num_params:,}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Best perplexity: {best_ppl:.4f}")
    print(f"  Tempo totale: {elapsed:.1f}s")
    print(f"  Log: {log_path}")
    print(f"{'='*60}\n")

    del model, optimizer, scaler
    torch.cuda.empty_cache()


# ==============================================================
# PLOT
# ==============================================================

def generate_plots():
    """Genera plot comparativi dai log JSON."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib non disponibile, skip plot.")
        return

    results_dir = _get_results_dir()

    # Carica tutti i log nanogpt
    logs = {}
    for f in sorted(os.listdir(results_dir)):
        if f.startswith("nanogpt_") and f.endswith(".json"):
            path = os.path.join(results_dir, f)
            with open(path) as fh:
                data = json.load(fh)
            scale = data.get("configurazione", {}).get("scale", "unknown")
            # Estrai attivazione dal nome
            parts = f.replace(".json", "").split("_")
            # nanogpt_{scale}_{activation}_{timestamp}
            if len(parts) >= 3:
                act = parts[2]
            else:
                continue
            key = (scale, act)
            # Tieni il log più recente per ogni (scala, attivazione)
            logs[key] = data

    if not logs:
        print("[PLOT] Nessun log nanogpt trovato.")
        return

    # Raccogli scale e attivazioni presenti
    scales_found = sorted(set(k[0] for k in logs.keys()),
                          key=lambda s: ["tiny", "small", "base"].index(s)
                          if s in ["tiny", "small", "base"] else 99)
    acts_found = sorted(set(k[1] for k in logs.keys()),
                        key=lambda a: ALL_ACTIVATIONS.index(a)
                        if a in ALL_ACTIVATIONS else 99)

    colors = {"nova": "#e74c3c", "gelu": "#3498db", "silu": "#2ecc71",
              "mish": "#9b59b6", "relu": "#f39c12"}

    # --- Plot 1: Training curves per scala ---
    fig, axes = plt.subplots(1, len(scales_found), figsize=(6 * len(scales_found), 5))
    if len(scales_found) == 1:
        axes = [axes]

    for ax, scale in zip(axes, scales_found):
        for act in acts_found:
            if (scale, act) not in logs:
                continue
            data = logs[(scale, act)]
            iters = [e["iter"] for e in data["iter_log"]]
            val_losses = [e["val_loss"] for e in data["iter_log"]]
            color = colors.get(act, "#333333")
            lw = 2.5 if act == "nova" else 1.5
            ax.plot(iters, val_losses, label=act.upper(), color=color, linewidth=lw)
        ax.set_title(f"NanoGPT-{scale.capitalize()}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Iterazione")
        ax.set_ylabel("Val Loss (CE)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("NanoGPT — Val Loss per Scala", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_nanogpt_training_curves.png"), dpi=150)
    plt.close()
    print("[PLOT] plot_nanogpt_training_curves.png salvato.")

    # --- Plot 2: Scaling curve (best val loss vs params) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for act in acts_found:
        params_list = []
        losses_list = []
        for scale in scales_found:
            if (scale, act) not in logs:
                continue
            data = logs[(scale, act)]
            params_list.append(data["configurazione"]["num_params"])
            losses_list.append(data["metriche"]["best_val_loss"])
        if params_list:
            color = colors.get(act, "#333333")
            lw = 2.5 if act == "nova" else 1.5
            ax.plot(params_list, losses_list, 'o-', label=act.upper(),
                    color=color, linewidth=lw, markersize=8)

    ax.set_xlabel("Parametri")
    ax.set_ylabel("Best Val Loss")
    ax.set_title("NanoGPT — Scaling Curve", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_nanogpt_scaling_curve.png"), dpi=150)
    plt.close()
    print("[PLOT] plot_nanogpt_scaling_curve.png salvato.")

    # --- Plot 3: Perplexity scaling ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for act in acts_found:
        params_list = []
        ppl_list = []
        for scale in scales_found:
            if (scale, act) not in logs:
                continue
            data = logs[(scale, act)]
            ppl = data["metriche"].get("best_perplexity")
            if ppl is not None:
                params_list.append(data["configurazione"]["num_params"])
                ppl_list.append(ppl)
        if params_list:
            color = colors.get(act, "#333333")
            lw = 2.5 if act == "nova" else 1.5
            ax.plot(params_list, ppl_list, 'o-', label=act.upper(),
                    color=color, linewidth=lw, markersize=8)

    ax.set_xlabel("Parametri")
    ax.set_ylabel("Best Perplexity")
    ax.set_title("NanoGPT — Perplexity vs Scala", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_nanogpt_perplexity_scaling.png"), dpi=150)
    plt.close()
    print("[PLOT] plot_nanogpt_perplexity_scaling.png salvato.")

    # --- Plot 4: Beta evolution (solo NOVA) ---
    has_beta = any(act == "nova" for _, act in logs.keys())
    if has_beta:
        fig, ax = plt.subplots(figsize=(8, 5))
        for scale in scales_found:
            if (scale, "nova") not in logs:
                continue
            data = logs[(scale, "nova")]
            iters = [e["iter"] for e in data["iter_log"] if "beta" in e]
            betas = [e["beta"] for e in data["iter_log"] if "beta" in e]
            if iters:
                ax.plot(iters, betas, 'o-', label=f"{scale.capitalize()}",
                        linewidth=2, markersize=5)

        ax.set_xlabel("Iterazione")
        ax.set_ylabel("β")
        ax.set_title("NanoGPT — Evoluzione di β (NOVA)", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "plot_nanogpt_beta_evolution.png"), dpi=150)
        plt.close()
        print("[PLOT] plot_nanogpt_beta_evolution.png salvato.")


# ==============================================================
# LAUNCHER: parallelo su 2 GPU
# ==============================================================

def launch_all():
    global _nova_cuda_ext

    # 1. Pre-download dataset
    print("[LAUNCHER] Pre-download TinyShakespeare...")
    data_root = _get_data_root()
    download_tinyshakespeare(data_root)
    print("[LAUNCHER] Dataset pronto.")

    # 2. Pre-compilazione kernel CUDA
    print("[LAUNCHER] Pre-compilazione kernel CUDA NOVA (30-60s)...")
    try:
        _nova_cuda_ext = _compile_nova_cuda()
        print("[LAUNCHER] Kernel CUDA compilato e cachato.")
    except Exception as e:
        print(f"[LAUNCHER] ATTENZIONE: Compilazione CUDA fallita: {e}")
        print("[LAUNCHER] NOVA userà il fallback Python puro.")

    # 3. Lancia esperimenti: per ogni scala, lancia attivazioni a coppie su 2 GPU
    script = os.path.abspath(__file__)

    for scale in ALL_SCALES:
        print(f"\n{'='*60}")
        print(f"  SCALA: {scale.upper()}")
        print(f"{'='*60}")

        # Lancia a coppie sulle 2 GPU
        pairs = []
        for i in range(0, len(ALL_ACTIVATIONS), 2):
            pair = [ALL_ACTIVATIONS[i]]
            if i + 1 < len(ALL_ACTIVATIONS):
                pair.append(ALL_ACTIVATIONS[i + 1])
            pairs.append(pair)

        for pair in pairs:
            procs = []
            for gpu, act in enumerate(pair):
                cmd = [sys.executable, script,
                       "--scale", scale, "--activation", act, "--gpu", str(gpu)]
                print(f"\n[LAUNCHER] Avvio: {' '.join(cmd)}")
                p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
                procs.append((act, p))

            for act, p in procs:
                p.wait()
                if p.returncode != 0:
                    print(f"[LAUNCHER] ERRORE: {scale}/{act} terminato con codice {p.returncode}")
                else:
                    print(f"[LAUNCHER] {scale.upper()}/{act.upper()} completato.")

    # 4. Genera plot
    print("\n[LAUNCHER] Generazione plot comparativi...")
    generate_plots()
    print("[LAUNCHER] Tutti gli esperimenti completati.")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NanoGPT TinyShakespeare: NOVA vs GELU/SiLU/Mish/ReLU")
    parser.add_argument("--scale", type=str,
                        choices=["tiny", "small", "base"],
                        help="Scala del modello (ometti per tutte)")
    parser.add_argument("--activation", type=str,
                        choices=ALL_ACTIVATIONS,
                        help="Funzione di attivazione (ometti per tutte)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="ID della GPU (default: 0)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Genera solo i plot dai log esistenti")
    args = parser.parse_args()

    if args.plot_only:
        generate_plots()
    elif args.scale is not None and args.activation is not None:
        run_experiment(args.scale, args.activation, args.gpu)
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            print(f"[ATTENZIONE] Solo {n_gpus} GPU disponibili.")
            print("Lancia manualmente con --scale, --activation e --gpu.")
            sys.exit(1)
        launch_all()
