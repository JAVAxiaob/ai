# -*- coding: utf-8 -*-
"""第三章：无监督学习。

包含：
1. K-means 聚类  kmeans_demo        - NumPy 手动实现
2. 主成分分析 PCA pca_demo           - NumPy 手动实现
3. 层次聚类       hierarchical_demo  - 使用 scipy / sklearn
"""

from __future__ import annotations

import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from utils import (
    ensure_dir, make_clustering_data, make_classification_data,
    plot_scatter_2d, plot_curve,
)


def _kmeans(X, k, max_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), k, replace=False)
    centers = X[idx].copy()
    for it in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(k)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return labels, centers


def kmeans_demo(k=3, output=None):
    """K-means 聚类：手动实现。"""
    X = make_clustering_data(n_samples=300, n_clusters=int(k), seed=10)
    labels, centers = _kmeans(X, k=int(k), max_iter=200, seed=11)
    print('簇中心：\\n', centers)

    out_path = None
    if output:
        out_path = os.path.join(output, 'ch03_kmeans.png')
        ensure_dir(out_path)
    plot_scatter_2d(X, labels, title='K-means (k=%d)' % int(k), output=out_path)
    return 'k=%d' % int(k)


def pca_demo(n_components=2, output=None):
    """主成分分析：手动实现。

    步骤：
    1. 对数据中心化
    2. 计算协方差矩阵 C = (1/(n-1)) * X.T @ X
    3. 对 C 特征值分解，取前 k 大特征值对应向量
    4. 投影：Z = X @ V
    """
    X, y = make_classification_data(n_samples=200, n_features=4, n_classes=3, seed=20)
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / (len(X) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    top_vecs = eigvecs[:, :int(n_components)]
    Z = X_centered @ top_vecs
    explained = eigvals / eigvals.sum()
    print('各主成分解释方差比：', explained[:4])
    print('投影后形状：', Z.shape)

    out_path = None
    if output:
        out_path = os.path.join(output, 'ch03_pca.png')
        ensure_dir(out_path)
        plot_scatter_2d(Z[:, :2], y, title='PCA projection', output=out_path)
    return 'explained_top=%.4f' % explained[:int(n_components)].sum()


def hierarchical_demo(k=3, output=None):
    """层次聚类：使用 scipy 的 linkage。"""
    from scipy.cluster.hierarchy import linkage, fcluster
    X = make_clustering_data(n_samples=150, n_clusters=int(k), seed=30)
    Z = linkage(X, method='ward')
    labels = fcluster(Z, t=int(k), criterion='maxclust')
    # fcluster 返回从 1 开始的簇号，转成 0 开始
    labels = labels - 1
    print('层次聚类：样本点分配到 %d 个簇' % int(k))

    out_path = None
    if output:
        out_path = os.path.join(output, 'ch03_hierarchical.png')
        ensure_dir(out_path)
        plot_scatter_2d(X, labels, title='Hierarchical Clustering', output=out_path)
    return 'k=%d' % int(k)
