# -*- coding: utf-8 -*-
"""第四章：深度学习基础。

包含：
1. 感知机        perceptron_demo  - 手动实现
2. 反向传播网络    backprop_demo    - 手动实现两层网络
3. 激活函数对比    activation_demo  - 可视化 Sigmoid / Tanh / ReLU
4. 损失函数对比    loss_demo        - MSE vs Cross-Entropy 可视化
"""

from __future__ import annotations

import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from utils import ensure_dir, make_classification_data, plot_scatter_2d, plot_curve


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def relu(x):
    return np.maximum(0, x)


def perceptron_demo(epochs=50, lr=0.1, output=None):
    """感知机：最简单的二分类线性模型。

    更新规则：若 y_i * (w*x_i + b) <= 0，则 w += lr*y_i*x_i, b += lr*y_i
    """
    X, y = make_classification_data(n_samples=100, n_features=2, n_classes=2, seed=0)
    y_signed = np.where(y == 1, 1, -1)
    w = np.zeros(2)
    b = 0.0
    losses = []
    for ep in range(epochs):
        mistakes = 0
        for xi, yi in zip(X, y_signed):
            if yi * (np.dot(w, xi) + b) <= 0:
                w += lr * yi * xi
                b += lr * yi
                mistakes += 1
        losses.append(mistakes)
        if ep % 10 == 0:
            print('感知机 epoch %d: 误分类样本数 = %d' % (ep, mistakes))

    preds = np.where(X @ w + b >= 0, 1, 0)
    acc = float(np.mean(preds == y))
    print('感知机训练准确率 =', acc)

    out_path = None
    if output:
        out_path = os.path.join(output, 'ch04_perceptron.png')
        ensure_dir(out_path)
        plot_scatter_2d(X, preds, title='Perceptron predictions', output=out_path)
    return 'accuracy=%.4f' % acc


def backprop_demo(epochs=500, lr=0.05, output=None):
    """一个简单的两层神经网络：输入 -> 隐藏层(ReLU) -> Sigmoid 输出。"""
    X, y = make_classification_data(n_samples=120, n_features=2, n_classes=2, seed=1)
    y = y.reshape(-1, 1).astype(float)
    rng = np.random.default_rng(42)
    hidden = 8
    W1 = rng.normal(0, 0.1, (2, hidden))
    b1 = np.zeros((1, hidden))
    W2 = rng.normal(0, 0.1, (hidden, 1))
    b2 = np.zeros((1, 1))
    m = len(X)
    loss_history = []
    for ep in range(epochs):
        # forward
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        # 二元交叉熵
        eps = 1e-7
        loss = float(-np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps)))
        loss_history.append(loss)
        # backward
        dz2 = (a2 - y) / m
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0).astype(float)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        if ep % 100 == 0:
            print('反向传播 epoch %d: loss=%.4f' % (ep, loss))

    preds = (a2 >= 0.5).astype(int).ravel()
    acc = float(np.mean(preds == y.ravel()))
    print('反向传播网络准确率 =', acc)

    if output:
        ensure_dir(output)
        out1 = os.path.join(output, 'ch04_backprop_loss.png')
        plot_curve([loss_history], ['loss'], title='Backprop Loss Curve', output=out1)
        out2 = os.path.join(output, 'ch04_backprop_pred.png')
        plot_scatter_2d(X, preds, title='Backprop NN predictions', output=out2)
    return 'accuracy=%.4f' % acc


def activation_demo(output=None):
    """可视化 Sigmoid / Tanh / ReLU 函数。"""
    x = np.linspace(-5, 5, 200)
    sig = sigmoid(x)
    tanh = np.tanh(x)
    rel = relu(x)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    plt.plot(x, sig, label='Sigmoid', linewidth=2)
    plt.plot(x, tanh, label='Tanh', linewidth=2)
    plt.plot(x, rel, label='ReLU', linewidth=2)
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.title('Activation Functions')
    plt.xlabel('x'); plt.ylabel('f(x)'); plt.legend(); plt.grid(True, alpha=0.3)

    out_path = None
    if output:
        out_path = os.path.join(output, 'ch04_activations.png')
        ensure_dir(out_path)
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        print('saved:', out_path)
    plt.close()
    return 'activation-visualized'


def loss_demo(output=None):
    """可视化 MSE 与二元交叉熵。"""
    p = np.linspace(0.01, 0.99, 200)
    # y=1 时
    bce1 = -np.log(p)
    mse1 = (1 - p) ** 2
    # y=0 时
    bce0 = -np.log(1 - p)
    mse0 = p ** 2

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(p, bce1, label='BCE (y=1)')
    axes[0].plot(p, mse1, label='MSE (y=1)')
    axes[0].set_title('y = 1')
    axes[0].set_xlabel('predicted p'); axes[0].set_ylabel('loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(p, bce0, label='BCE (y=0)')
    axes[1].plot(p, mse0, label='MSE (y=0)')
    axes[1].set_title('y = 0')
    axes[1].set_xlabel('predicted p'); axes[1].set_ylabel('loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.suptitle('Loss Functions')

    out_path = None
    if output:
        out_path = os.path.join(output, 'ch04_losses.png')
        ensure_dir(out_path)
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        print('saved:', out_path)
    plt.close()
    return 'loss-visualized'
