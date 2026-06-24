# -*- coding: utf-8 -*-
"""第2章：采样量化、邻域、距离、卷积演示。"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from utils import ImageArray, get_kernel_size, to_gray


def downsample(img: ImageArray, *params) -> ImageArray:
    """降低空间分辨率（采样）。参数: factor (默认 4)"""
    f = int(params[0]) if params else 4
    f = max(2, f)
    small = img[::f, ::f] if img.ndim == 2 else img[::f, ::f, :]
    h, w = img.shape[:2]
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def quantize_levels(img: ImageArray, *params) -> ImageArray:
    """灰度量化。参数: bits (默认 4，即 16 级)"""
    bits = int(params[0]) if params else 4
    levels = 2 ** max(1, min(bits, 8))
    gray = to_gray(img)
    step = 256 // levels
    return ((gray // step) * step).astype(np.uint8)


def mark_neighbors(img: ImageArray, *params) -> ImageArray:
    """
    标记 4/8 邻域中心像素。
    参数: row, col[, connectivity=8]
    """
    r = int(params[0]) if len(params) > 0 else img.shape[0] // 2
    c = int(params[1]) if len(params) > 1 else img.shape[1] // 2
    conn = int(params[2]) if len(params) > 2 else 8
    out = cv2.cvtColor(to_gray(img), cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    r, c = min(max(0, r), h - 1), min(max(0, c), w - 1)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if conn == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    cv2.circle(out, (c, r), 3, (0, 0, 255), -1)
    for dr, dc in offsets:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            cv2.circle(out, (nc, nr), 2, (0, 255, 0), -1)
    return out


def distance_transform_vis(img: ImageArray, *params) -> ImageArray:
    """
    距离变换可视化（欧氏/街区/棋盘由 metric 决定）。
    参数: metric (euclidean|cityblock|chessboard，默认 euclidean)
    """
    metric = str(params[0]).lower() if params else "euclidean"
    gray = to_gray(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    mask = (binary > 0).astype(np.uint8)
    if metric == "cityblock":
        dist = cv2.distanceTransform(mask, cv2.DIST_L1, 3)
    elif metric in ("chessboard", "chess"):
        dist = cv2.distanceTransform(mask, cv2.DIST_C, 3)
    else:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
    return dist_norm.astype(np.uint8)


def convolve_demo(img: ImageArray, *params) -> ImageArray:
    """矩阵卷积演示（平滑核）。参数: kernel_size (默认 3)"""
    k = get_kernel_size(params, 3)
    kernel = np.ones((k, k), np.float64) / (k * k)
    gray = to_gray(img).astype(np.float64)
    out = ndimage.convolve(gray, kernel, mode="nearest")
    return np.clip(out, 0, 255).astype(np.uint8)


def add_gaussian_noise(img: ImageArray, *params) -> ImageArray:
    """添加高斯噪声（量化噪声模拟）。参数: sigma (默认 25)"""
    sigma = float(params[0]) if params else 25.0
    gray = to_gray(img).astype(np.float64)
    noise = np.random.default_rng(42).normal(0, sigma, gray.shape)
    return np.clip(gray + noise, 0, 255).astype(np.uint8)


def connectivity_components_vis(img: ImageArray, *params) -> ImageArray:
    """连通分量标记可视化（4/8 连通）。参数: connectivity (4|8，默认 8)"""
    conn = int(params[0]) if params else 8
    gray = to_gray(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    connectivity = 8 if conn == 8 else 4
    num, labels = cv2.connectedComponents(binary, connectivity=connectivity)
    if num <= 1:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    label_hue = np.uint8(179 * labels / max(num - 1, 1))
    colored = cv2.applyColorMap(label_hue, cv2.COLORMAP_HSV)
    colored[labels == 0] = 0
    return colored


def optical_illusion_demo(img: ImageArray, *params) -> ImageArray:
    """视觉错觉演示：在输入图上叠加赫尔曼网格错觉图案。"""
    h, w = to_gray(img).shape
    tile = int(params[0]) if params else 40
    grid = np.ones((h, w), np.uint8) * 128
    for i in range(0, h, tile):
        for j in range(0, w, tile):
            cv2.rectangle(grid, (j, i), (min(j + tile, w), min(i + tile, h)), 255, -1)
            cv2.circle(grid, (j + tile // 2, i + tile // 2), max(2, tile // 10), 0, -1)
    base = to_gray(img)
    blend = cv2.addWeighted(base, 0.6, grid, 0.4, 0)
    return cv2.cvtColor(blend, cv2.COLOR_GRAY2BGR)


def probability_noise_demo(img: ImageArray, *params) -> ImageArray:
    """
    概率与随机过程演示：对图像施加高斯噪声并显示灰度直方图对比条。
    参数: sigma (默认 20)
    """
    sigma = float(params[0]) if params else 20.0
    gray = to_gray(img)
    rng = np.random.default_rng(42)
    noisy = np.clip(gray.astype(np.float64) + rng.normal(0, sigma, gray.shape), 0, 255).astype(
        np.uint8
    )
    hist_orig = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist_noisy = cv2.calcHist([noisy], [0], None, [256], [0, 256]).ravel()
    h_bar = 64
    bar_w = min(256, gray.shape[1])
    ho = (hist_orig / (hist_orig.max() + 1e-6) * 255).astype(np.uint8)
    hn = (hist_noisy / (hist_noisy.max() + 1e-6) * 255).astype(np.uint8)
    bar = np.zeros((h_bar, bar_w, 3), np.uint8)
    for x in range(bar_w):
        h0 = int(min(h_bar, ho[x] * h_bar // 255))
        h1 = int(min(h_bar, hn[x] * h_bar // 255))
        if h0 > 0:
            bar[h_bar - h0 :, x, 1] = 255
        if h1 > 0:
            bar[h_bar - h1 :, x, 2] = 255
    bar = cv2.resize(bar, (gray.shape[1], h_bar))
    top = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
    return np.vstack([top, bar])


def spectrum_band_simulation(img: ImageArray, *params) -> ImageArray:
    """
    模拟不同波段成像（可见光/红外/雷达风格）。
    参数: band (visible|infrared|radar，默认 visible)
    """
    band = str(params[0]).lower() if params else "visible"
    gray = to_gray(img).astype(np.float64)
    if band in ("infrared", "ir"):
        out = cv2.GaussianBlur(gray, (9, 9), 0) * 1.2
    elif band in ("radar", "xray", "x"):
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        out = np.sqrt(gx**2 + gy**2)
    else:
        out = gray
    return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
