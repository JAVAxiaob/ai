# -*- coding: utf-8 -*-
"""通用工具函数：数据集生成、可视化、参数解析等。"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Any, List, Tuple


def make_regression_data(n_samples=100, noise=5.0, seed=42):
    """生成线性回归示例数据 y = 2.5*x + 3 + 噪声。"""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10, 10, n_samples)
    y = 2.5 * x + 3.0 + rng.normal(0, noise, n_samples)
    return x.reshape(-1, 1), y


def make_classification_data(n_samples=200, n_features=2, n_classes=2, seed=42):
    """生成简单的分类示例数据（高斯团簇）。"""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    centers = rng.uniform(-3, 3, (n_classes, n_features))
    for i in range(n_classes):
        pts = centers[i] + rng.normal(0, 0.8, (n_samples // n_classes, n_features))
        xs.append(pts)
        ys.append(np.full(len(pts), i))
    X = np.vstack(xs)
    y = np.concatenate(ys)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_clustering_data(n_samples=300, n_clusters=3, seed=42):
    """生成用于聚类演示的数据。"""
    return make_classification_data(n_samples, 2, n_clusters, seed)[0]


def plot_scatter_2d(X, y=None, title='Scatter', output=None):
    """绘制 2D 散点图，支持类别颜色。"""
    plt.figure(figsize=(6, 5))
    if y is None:
        plt.scatter(X[:, 0], X[:, 1], alpha=0.7, s=30)
    else:
        for c in np.unique(y):
            mask = y == c
            plt.scatter(X[mask, 0], X[mask, 1], label='class %d' % int(c), alpha=0.7, s=30)
        plt.legend()
    plt.title(title); plt.xlabel('x1'); plt.ylabel('x2'); plt.grid(True, alpha=0.3)
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or '.', exist_ok=True)
        plt.savefig(output, dpi=100, bbox_inches='tight')
        print('saved: ' + output)
    plt.close()


def plot_regression_fit(x, y, y_pred, title='Regression', output=None):
    """绘制散点 + 拟合曲线。"""
    plt.figure(figsize=(6, 5))
    plt.scatter(x.ravel(), y, s=25, alpha=0.6, label='samples', color='#3366CC')
    order = np.argsort(x.ravel())
    plt.plot(x.ravel()[order], y_pred[order], color='#CC3333', linewidth=2.0, label='fit')
    plt.title(title); plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True, alpha=0.3)
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or '.', exist_ok=True)
        plt.savefig(output, dpi=100, bbox_inches='tight')
        print('saved: ' + output)
    plt.close()


def plot_curve(xs_list, labels, title='Curve', output=None):
    """绘制多条曲线，用于训练过程可视化。"""
    plt.figure(figsize=(6, 4))
    for x, lab in zip(xs_list, labels):
        plt.plot(list(x), label=lab)
    plt.title(title); plt.legend(); plt.grid(True, alpha=0.3)
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or '.', exist_ok=True)
        plt.savefig(output, dpi=100, bbox_inches='tight')
        print('saved: ' + output)
    plt.close()


def parse_params(param_value):
    """解析字符串 '1,2,3' -> [1.0, 2.0, 3.0]。"""
    if param_value is None or param_value == '':
        return []
    if isinstance(param_value, (list, tuple)):
        return list(param_value)
    s = str(param_value).strip()
    parts = [p.strip() for p in s.split(',')]
    out = []
    for p in parts:
        try:
            out.append(float(p) if '.' in p else int(p))
        except ValueError:
            out.append(p)
    return out


def accuracy(y_true, y_pred):
    """计算分类准确率。"""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def mse(y_true, y_pred):
    """均方误差。"""
    return float(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2))


def train_test_split(X, y, test_size=0.3, seed=42):
    """极简版 train_test_split。"""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    split = int(n * (1 - test_size))
    tr, te = idx[:split], idx[split:]
    return X[tr], X[te], y[tr], y[te]


def ensure_dir(path):
    """确保目录存在。"""
    if os.path.basename(path) and '.' in os.path.basename(path):
        d = os.path.dirname(os.path.abspath(path))
    else:
        d = os.path.abspath(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
