# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ==========================================
# 2. IMPLEMENTAZIONE KERNEL CUDA (FORWARD & BACKWARD)
# ==========================================
cpp_source = """
torch::Tensor nova_cuda_forward(torch::Tensor x, float beta);
std::vector<torch::Tensor> nova_cuda_backward(torch::Tensor grad_output, torch::Tensor x, float beta);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// ===================== FORWARD KERNEL =====================
// f(x) = x * sigmoid(beta * x) - x / (1 + (beta * x)^2)
// Tutte le operazioni intermedie in float32 per stabilità numerica
// con FP16/BF16.

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

// ===================== BACKWARD KERNEL =====================
// Calcola sia grad_input (df/dx) sia grad_beta per-elemento (df/dbeta).
//
// df/dx = sigmoid(bx) + bx * sigmoid'(bx) - (1 - (bx)^2) / (1 + (bx)^2)^2
//
// df/dbeta = x^2 * sigmoid(bx) * (1 - sigmoid(bx)) + 2*beta*x^3 / (1 + (bx)^2)^2

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

        // --- grad w.r.t. x ---
        const float d_gating_dx = sig + bx * sig_deriv;
        const float bx_sq = bx * bx;
        const float denom = 1.0f + bx_sq;
        const float d_rational_dx = (1.0f - bx_sq) / (denom * denom);
        grad_input[index] = static_cast<scalar_t>(go * (d_gating_dx - d_rational_dx));

        // --- grad w.r.t. beta ---
        const float val_sq = val * val;
        const float d_gating_dbeta = val_sq * sig_deriv;
        const float d_rational_dbeta = -2.0f * beta * val_sq * val / (denom * denom);
        grad_beta_elem[index] = static_cast<scalar_t>(go * (d_gating_dbeta - d_rational_dbeta));
    }
}

// ===================== C++ WRAPPERS =====================

torch::Tensor nova_cuda_forward(torch::Tensor x, float beta) {
    auto out = torch::empty_like(x);
    const int threads = 256;
    const int blocks = (x.numel() + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "nova_forward_cuda", ([&] {
            nova_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), beta, x.numel());
        }));
    return out;
}

std::vector<torch::Tensor> nova_cuda_backward(torch::Tensor grad_output, torch::Tensor x, float beta) {
    auto grad_input = torch::empty_like(x);
    auto grad_beta_elem = torch::empty_like(x);
    const int threads = 256;
    const int blocks = (x.numel() + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "nova_backward_cuda", ([&] {
            nova_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                grad_output.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(), grad_beta_elem.data_ptr<scalar_t>(),
                beta, x.numel());
        }));
    // Riduzione: somma per-elemento -> scalare per il gradiente di beta
    auto grad_beta = grad_beta_elem.sum();
    return {grad_input, grad_beta};
}
"""

def compile_nova_cuda():
    print("   [JIT Compiler] Compilazione Kernel CUDA in corso (potrebbe richiedere 30-60s)...")
    try:
        nova_ext = load_inline(
            name='nova_cuda_ext',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=['nova_cuda_forward', 'nova_cuda_backward'],
            with_cuda=True,
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '--use_fast_math']
        )
        print("   [JIT Compiler] Compilazione completata con successo!")
        return nova_ext
    except Exception as e:
        print(f"   [!] Compilazione fallita, fallback su Python Puro: {e}")
        return None

class NOVAFunction(torch.autograd.Function):
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
        grad_x, grad_beta = ctx.nova_ext.nova_cuda_backward(grad_output, x, ctx.beta_val)
        return grad_x, grad_beta, None

class NOVAFusedCUDA(nn.Module):
    def __init__(self, nova_ext, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.nova_ext = nova_ext
    def forward(self, x):
        return NOVAFunction.apply(x, self.beta, self.nova_ext)
