# -*- coding: utf-8 -*-
"""第6章：彩色模型转换、增强、滤波、分割。"""

from __future__ import annotations

import cv2
import numpy as np

from utils import ImageArray, to_gray


def _ensure_bgr(img: ImageArray) -> ImageArray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def rgb_to_hsv(img: ImageArray, *params) -> ImageArray:
    """RGB(BGR) 转 HSV 并拼接显示通道。"""
    bgr = _ensure_bgr(img)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return np.hstack([h, s, v])


def rgb_to_hsi(img: ImageArray, *params) -> ImageArray:
    """RGB 转 HSI 并拼接显示（OpenCV 用 HSV 近似 HSI 的 H/S/I 通道）。"""
    bgr = _ensure_bgr(img)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    intensity = v
    return np.hstack([h, s, intensity])


def rgb_to_cmy(img: ImageArray, *params) -> ImageArray:
    """BGR 转 CMY（C=255-B, M=255-G, Y=255-R）并水平拼接。"""
    bgr = _ensure_bgr(img).astype(np.float64)
    b, g, r = cv2.split(bgr)
    c = 255.0 - r
    m = 255.0 - g
    y = 255.0 - b
    return np.hstack(
        [c.astype(np.uint8), m.astype(np.uint8), y.astype(np.uint8)]
    )


def rgb_to_cmyk(img: ImageArray, *params) -> ImageArray:
    """BGR 转 CMYK 并水平拼接四通道（K 为黑色分量）。"""
    bgr = _ensure_bgr(img).astype(np.float64) / 255.0
    b, g, r = cv2.split(bgr)
    k = 1.0 - np.maximum(np.maximum(r, g), b)
    denom = np.maximum(1.0 - k, 1e-6)
    c = np.clip((1.0 - r - k) / denom, 0, 1)
    m = np.clip((1.0 - g - k) / denom, 0, 1)
    y = np.clip((1.0 - b - k) / denom, 0, 1)
    channels = [
        (c * 255).astype(np.uint8),
        (m * 255).astype(np.uint8),
        (y * 255).astype(np.uint8),
        (k * 255).astype(np.uint8),
    ]
    return np.hstack(channels)


def rgb_to_ycbcr(img: ImageArray, *params) -> ImageArray:
    """转 YCbCr 并水平拼接。"""
    bgr = _ensure_bgr(img)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return np.hstack(cv2.split(ycc))


def color_to_gray(img: ImageArray, *params) -> ImageArray:
    """彩色转灰度。参数: method (weighted|avg|max，默认 weighted)"""
    bgr = _ensure_bgr(img)
    method = str(params[0]).lower() if params else "weighted"
    if method == "avg":
        return np.mean(bgr, axis=2).astype(np.uint8)
    if method == "max":
        return np.max(bgr, axis=2).astype(np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def color_contrast_enhance(img: ImageArray, *params) -> ImageArray:
    """在 HSV 空间增强 V 通道。参数: scale (默认 1.3)"""
    scale = float(params[0]) if params else 1.3
    bgr = _ensure_bgr(img)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def pseudocolor(img: ImageArray, *params) -> ImageArray:
    """伪彩色（应用 colormap）。参数: colormap (默认 jet)"""
    cmap_name = str(params[0]).lower() if params else "jet"
    gray = to_gray(img)
    cmap = getattr(cv2, f"COLORMAP_{cmap_name.upper()}", cv2.COLORMAP_JET)
    return cv2.applyColorMap(gray, cmap)


def color_smooth(img: ImageArray, *params) -> ImageArray:
    """彩色平滑。参数: kernel_size (默认 5)"""
    k = int(params[0]) if params else 5
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(_ensure_bgr(img), (k, k), 0)


def color_sharpen(img: ImageArray, *params) -> ImageArray:
    """彩色锐化。参数: amount (默认 1.0)"""
    amount = float(params[0]) if params else 1.0
    bgr = _ensure_bgr(img).astype(np.float64)
    blur = cv2.GaussianBlur(bgr, (0, 0), 3)
    out = bgr + amount * (bgr - blur)
    return np.clip(out, 0, 255).astype(np.uint8)


def color_edge_canny(img: ImageArray, *params) -> ImageArray:
    """彩色边缘（各通道 Canny 合并）。参数: low, high (默认 50, 150)"""
    low = int(params[0]) if len(params) > 0 else 50
    high = int(params[1]) if len(params) > 1 else 150
    bgr = _ensure_bgr(img)
    edges = [cv2.Canny(bgr[:, :, i], low, high) for i in range(3)]
    return np.maximum.reduce(edges)


def color_threshold_segment(img: ImageArray, *params) -> ImageArray:
    """HSV 颜色阈值分割。参数: h_low, h_high, s_low, s_high, v_low, v_high"""
    h_lo = int(params[0]) if len(params) > 0 else 0
    h_hi = int(params[1]) if len(params) > 1 else 180
    s_lo = int(params[2]) if len(params) > 2 else 50
    s_hi = int(params[3]) if len(params) > 3 else 255
    v_lo = int(params[4]) if len(params) > 4 else 50
    v_hi = int(params[5]) if len(params) > 5 else 255
    hsv = cv2.cvtColor(_ensure_bgr(img), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))
    return cv2.bitwise_and(_ensure_bgr(img), _ensure_bgr(img), mask=mask)


def color_region_growing(img: ImageArray, *params) -> ImageArray:
    """彩色区域生长（种子点）。参数: row, col, threshold (默认中心, 25)"""
    bgr = _ensure_bgr(img)
    h, w = bgr.shape[:2]
    sr = int(params[0]) if len(params) > 0 else h // 2
    sc = int(params[1]) if len(params) > 1 else w // 2
    th = float(params[2]) if len(params) > 2 else 25.0
    seed = bgr[sr, sc].astype(np.float64)
    diff = np.linalg.norm(bgr.astype(np.float64) - seed, axis=2)
    mask = (diff < th).astype(np.uint8) * 255
    return cv2.bitwise_and(bgr, bgr, mask=mask)
