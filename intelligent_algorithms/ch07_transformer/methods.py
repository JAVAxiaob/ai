# -*- coding: utf-8 -*-
"""第七章：Transformer 与注意力机制。

包含：
1. 缩放点积注意力 attention_demo - NumPy 手动实现
2. Transformer 前向传播 (简化) transformer_demo
"""

from __future__ import annotations

import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from utils import ensure_dir


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    """缩放点积注意力：Att(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V。"""
    d_k = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    attn = _softmax(scores, axis=-1)
    out = np.matmul(attn, v)
    return out, attn


def attention_demo(output=None):
    """可视化一个 4x4 的注意力矩阵。"""
    rng = np.random.default_rng(0)
    seq_len = 6
    d_k = 8
    q = rng.normal(0, 0.5, (seq_len, d_k))
    k = rng.normal(0, 0.5, (seq_len, d_k))
    v = rng.normal(0, 0.5, (seq_len, d_k))
    out, attn = scaled_dot_product_attention(q, k, v)
    print('输出形状:', out.shape)
    print('注意力权重形状:', attn.shape)
    print('每一行和为 1?', np.allclose(attn.sum(axis=-1), 1.0))

    if output:
        ensure_dir(output)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.imshow(attn, cmap='Blues', vmin=0, vmax=1)
        for i in range(seq_len):
            for j in range(seq_len):
                plt.text(j, i, '%.2f' % attn[i, j], ha='center', va='center', fontsize=8, color='black')
        plt.title('Attention weights')
        plt.xlabel('key pos'); plt.ylabel('query pos')
        fp = os.path.join(output, 'ch07_attention.png')
        plt.savefig(fp, dpi=100, bbox_inches='tight')
        print('saved:', fp)
        plt.close()
    return 'attention-done'


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, rng):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.Wq = rng.normal(0, 0.1, (d_model, d_model))
        self.Wk = rng.normal(0, 0.1, (d_model, d_model))
        self.Wv = rng.normal(0, 0.1, (d_model, d_model))
        self.Wo = rng.normal(0, 0.1, (d_model, d_model))

    def forward(self, x):
        # x: (seq_len, d_model)
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv
        # split into heads
        q = q.reshape(-1, self.h, self.d_k).transpose(1, 0, 2)
        k = k.reshape(-1, self.h, self.d_k).transpose(1, 0, 2)
        v = v.reshape(-1, self.h, self.d_k).transpose(1, 0, 2)
        out, _ = scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 0, 2).reshape(-1, self.d_model)
        return out @ self.Wo


def transformer_demo(d_model=16, num_heads=4, seq_len=8, output=None):
    """简化 Transformer 前向：多头注意力 + 前馈 + 残差。"""
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.5, (seq_len, d_model))
    mha = MultiHeadAttention(d_model, num_heads, rng)
    attn_out = mha.forward(x)
    # 残差 + 层归一化
    x2 = x + attn_out
    x2 = (x2 - x2.mean(axis=-1, keepdims=True)) / (x2.std(axis=-1, keepdims=True) + 1e-6)
    # 前馈
    ff_w1 = rng.normal(0, 0.1, (d_model, d_model * 2))
    ff_b1 = np.zeros(d_model * 2)
    ff_w2 = rng.normal(0, 0.1, (d_model * 2, d_model))
    ff_b2 = np.zeros(d_model)
    hidden = np.maximum(0, x2 @ ff_w1 + ff_b1)
    out = hidden @ ff_w2 + ff_b2
    # 第二次残差
    out = x2 + out
    out = (out - out.mean(axis=-1, keepdims=True)) / (out.std(axis=-1, keepdims=True) + 1e-6)
    print('Transformer 输出形状:', out.shape)

    if output:
        ensure_dir(output)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        plt.imshow(out, aspect='auto', cmap='viridis')
        plt.title('Transformer output (seq x d_model)')
        plt.xlabel('d_model'); plt.ylabel('position')
        fp = os.path.join(output, 'ch07_transformer.png')
        plt.savefig(fp, dpi=100, bbox_inches='tight')
        print('saved:', fp)
        plt.close()
    return 'transformer-done'
