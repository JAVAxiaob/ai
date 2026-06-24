# -*- coding: utf-8 -*-
"""第八章：Voyo 算法。

Voyo 算法是一个教学用的自定义优化算法（取自汉语拼音 "优化" 的谐音），
用于演示 "动量 + 自适应学习率" 的思想。

核心公式：
    m_t   = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t   = beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    theta = theta - lr * m_hat / (sqrt(v_hat) + eps)

本章节对简单的二次函数做优化并可视化收敛轨迹。
"""

from __future__ import annotations

import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from utils import ensure_dir


def _f(theta):
    """目标函数：f(x, y) = (x-2)^2 + 0.5*(y+1)^2 + 3。"""
    x, y = theta
    return (x - 2.0) ** 2 + 0.5 * (y + 1.0) ** 2 + 3.0


def _grad(theta):
    x, y = theta
    return np.array([2.0 * (x - 2.0), 1.0 * (y + 1.0)])


def voyo_optimize(theta0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, epochs=200):
    """Voyo 优化（教学用 Adam 风格算法）。"""
    theta = np.array(theta0, dtype=float).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history = [theta.copy()]
    losses = [_f(theta)]
    for t in range(1, epochs + 1):
        g = _grad(theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(theta.copy())
        losses.append(_f(theta))
        if t % 40 == 0:
            print('Voyo step %d: theta=%s loss=%.4f' % (t, str(theta), losses[-1]))
    return theta, np.array(history), losses


def voyo_demo(output=None):
    """Voyo 算法：二次函数优化 + 等高线轨迹图。"""
    theta0 = [-4.0, 4.0]
    best, history, losses = voyo_optimize(theta0, lr=0.1, epochs=200)
    print('Voyo 最终解:', best)
    print('理论最优解: [2.0, -1.0]')

    if output:
        ensure_dir(output)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # 等高线 + 轨迹
        x = np.linspace(-6, 6, 100)
        y = np.linspace(-6, 6, 100)
        X, Y = np.meshgrid(x, y)
        Z = (X - 2.0) ** 2 + 0.5 * (Y + 1.0) ** 2 + 3.0
        axes[0].contour(X, Y, Z, levels=30, cmap='viridis')
        axes[0].plot(history[:, 0], history[:, 1], 'o-', color='red', markersize=3, label='trajectory')
        axes[0].scatter([2.0], [-1.0], marker='*', color='yellow', s=200, label='optimum')
        axes[0].set_title('Voyo trajectory on contour')
        axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(losses)
        axes[1].set_title('Voyo loss curve')
        axes[1].set_xlabel('step'); axes[1].set_ylabel('f(theta)')
        axes[1].grid(True, alpha=0.3)
        fp = os.path.join(output, 'ch08_voyo.png')
        plt.savefig(fp, dpi=100, bbox_inches='tight')
        print('saved:', fp)
        plt.close()
    return 'best=[%.3f,%.3f]' % (best[0], best[1])
