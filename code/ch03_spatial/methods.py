# -*- coding: utf-8 -*-
"""第3章：灰度变换、直方图、空间滤波、模糊集增强。"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from utils import ImageArray, ensure_float, get_kernel_size, to_gray


def linear_transform(img: ImageArray, *params) -> ImageArray:
    """线性变换 g=af+b。参数: a, b (默认 1.2, 10)"""
    a = float(params[0]) if len(params) > 0 else 1.2
    b = float(params[1]) if len(params) > 1 else 10.0
    gray = ensure_float(to_gray(img))
    return np.clip(a * gray + b, 0, 255).astype(np.uint8)


def contrast_stretch(img: ImageArray, *params) -> ImageArray:
    """对比度拉伸。参数: low_percent, high_percent (默认 2, 98)"""
    p_low = float(params[0]) if len(params) > 0 else 2.0
    p_high = float(params[1]) if len(params) > 1 else 98.0
    gray = to_gray(img)
    lo, hi = np.percentile(gray, (p_low, p_high))
    if hi <= lo:
        return gray
    out = (gray.astype(np.float64) - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def log_transform(img: ImageArray, *params) -> ImageArray:
    """对数变换。参数: c (默认 1.0)"""
    c = float(params[0]) if params else 1.0
    gray = ensure_float(to_gray(img))
    gray = np.maximum(gray, 1.0)
    out = c * np.log1p(gray) / np.log1p(255.0) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def power_law_transform(img: ImageArray, *params) -> ImageArray:
    """幂律（伽马）变换。参数: gamma (默认 0.5)"""
    gamma = float(params[0]) if params else 0.5
    gray = to_gray(img).astype(np.float64) / 255.0
    out = np.power(gray, gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def threshold_global(img: ImageArray, *params) -> ImageArray:
    """全局阈值化。参数: thresh (默认 Otsu 用 0 表示自动)"""
    gray = to_gray(img)
    if params and int(params[0]) > 0:
        t = int(params[0])
    else:
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, out = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return out


def histogram_equalization(img: ImageArray, *params) -> ImageArray:
    """直方图均衡化。"""
    gray = to_gray(img)
    return cv2.equalizeHist(gray)


def histogram_matching(img: ImageArray, *params) -> ImageArray:
    """直方图规定化（匹配参考图路径或默认均匀）。"""
    from skimage import exposure

    gray = to_gray(img)
    if params:
        ref = cv2.imread(str(params[0]), cv2.IMREAD_GRAYSCALE)
        if ref is not None:
            return (exposure.match_histograms(gray, ref) * 255).astype(np.uint8)
    return (exposure.equalize_hist(gray) * 255).astype(np.uint8)


def local_histogram_equalization(img: ImageArray, *params) -> ImageArray:
    """局部直方图均衡（CLAHE）。参数: clip_limit, tile_size (默认 2.0, 8)"""
    clip = float(params[0]) if len(params) > 0 else 2.0
    tile = int(params[1]) if len(params) > 1 else 8
    gray = to_gray(img)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def mean_filter(img: ImageArray, *params) -> ImageArray:
    """均值平滑。参数: kernel_size (默认 5)"""
    k = get_kernel_size(params, 5)
    gray = to_gray(img)
    return cv2.blur(gray, (k, k))


def gaussian_filter(img: ImageArray, *params) -> ImageArray:
    """高斯平滑。参数: kernel_size, sigma (默认 5, 1.0)"""
    k = get_kernel_size(params, 5)
    sigma = float(params[1]) if len(params) > 1 else 1.0
    gray = to_gray(img)
    return cv2.GaussianBlur(gray, (k, k), sigma)


def median_filter(img: ImageArray, *params) -> ImageArray:
    """中值滤波。参数: kernel_size (默认 5)"""
    k = get_kernel_size(params, 5)
    gray = to_gray(img)
    return cv2.medianBlur(gray, k)


def sobel_gradient(img: ImageArray, *params) -> ImageArray:
    """Sobel 梯度幅值。参数: ksize (默认 3)"""
    k = int(params[0]) if params else 3
    gray = to_gray(img)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def laplacian_sharpen(img: ImageArray, *params) -> ImageArray:
    """拉普拉斯锐化。参数: ksize (默认 3)"""
    k = int(params[0]) if params else 3
    gray = to_gray(img).astype(np.float64)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
    out = gray - lap
    return np.clip(out, 0, 255).astype(np.uint8)


def highpass_filter(img: ImageArray, *params) -> ImageArray:
    """高通滤波（原图减低通）。参数: kernel_size (默认 15)"""
    k = get_kernel_size(params, 15)
    gray = to_gray(img).astype(np.float64)
    low = cv2.GaussianBlur(gray, (k, k), 0)
    high = gray - low + 128
    return np.clip(high, 0, 255).astype(np.uint8)


def unsharp_mask(img: ImageArray, *params) -> ImageArray:
    """非锐化掩模。参数: amount, sigma (默认 1.5, 3.0)"""
    amount = float(params[0]) if len(params) > 0 else 1.5
    sigma = float(params[1]) if len(params) > 1 else 3.0
    gray = to_gray(img).astype(np.float64)
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    out = gray + amount * (gray - blur)
    return np.clip(out, 0, 255).astype(np.uint8)


def fuzzy_enhancement(img: ImageArray, *params) -> ImageArray:
    """
    模糊集空间增强：低/中/高灰度隶属度加权。
    参数: crossover_low, crossover_high (默认 85, 170)
    """
    c1 = float(params[0]) if len(params) > 0 else 85.0
    c2 = float(params[1]) if len(params) > 1 else 170.0
    gray = ensure_float(to_gray(img)) / 255.0
    dark = np.exp(-((gray - 0.0) ** 2) / (2 * (c1 / 255.0) ** 2 + 1e-6))
    bright = np.exp(-((gray - 1.0) ** 2) / (2 * ((255 - c2) / 255.0) ** 2 + 1e-6))
    mid = 1.0 - dark - bright
    enhanced = dark * (gray * 0.5) + mid * gray + bright * (0.5 + gray * 0.5)
    return (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
