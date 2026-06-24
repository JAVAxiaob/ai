# -*- coding: utf-8 -*-
"""第4章：DFT、理想/巴特沃斯/高斯滤波、同态滤波。"""

from __future__ import annotations

import cv2
import numpy as np

from utils import ImageArray, to_gray


def _dft_shift(gray: np.ndarray) -> tuple:
    """二维 DFT 并中心化。"""
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    return f, fshift


def dft_spectrum(img: ImageArray, *params) -> ImageArray:
    """显示频谱幅度（对数）。"""
    gray = to_gray(img)
    _, fshift = _dft_shift(gray)
    magnitude = 20 * np.log1p(np.abs(fshift))
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _filter_mask(shape: tuple, d0: float, kind: str, order: int = 2) -> np.ndarray:
    """构造频域滤波器掩模。"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u, v = np.ogrid[:rows, :cols]
    d = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
    if kind == "ideal_low":
        return (d <= d0).astype(np.float64)
    if kind == "ideal_high":
        return (d > d0).astype(np.float64)
    if kind == "butterworth_low":
        return 1.0 / (1.0 + (d / (d0 + 1e-6)) ** (2 * order))
    if kind == "butterworth_high":
        return 1.0 / (1.0 + (d0 / (d + 1e-6)) ** (2 * order))
    if kind == "gaussian_low":
        return np.exp(-(d**2) / (2 * (d0**2) + 1e-6))
    if kind == "gaussian_high":
        return 1.0 - np.exp(-(d**2) / (2 * (d0**2) + 1e-6))
    if kind == "bandpass":
        w = d0 * 0.5
        outer = d0 + w
        inner = max(d0 - w, 1)
        return ((d >= inner) & (d <= outer)).astype(np.float64)
    if kind == "bandreject":
        w = d0 * 0.5
        outer = d0 + w
        inner = max(d0 - w, 1)
        return ((d < inner) | (d > outer)).astype(np.float64)
    return np.ones(shape, dtype=np.float64)


def _apply_freq_filter(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    f, fshift = _dft_shift(gray)
    filtered = fshift * mask
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    return np.clip(img_back, 0, 255).astype(np.uint8)


def ideal_lowpass(img: ImageArray, *params) -> ImageArray:
    """理想低通。参数: D0 (默认 30)"""
    d0 = float(params[0]) if params else 30.0
    gray = to_gray(img)
    mask = _filter_mask(gray.shape, d0, "ideal_low")
    return _apply_freq_filter(gray, mask)


def ideal_highpass(img: ImageArray, *params) -> ImageArray:
    """理想高通。参数: D0 (默认 30)"""
    d0 = float(params[0]) if params else 30.0
    gray = to_gray(img)
    mask = _filter_mask(gray.shape, d0, "ideal_high")
    return _apply_freq_filter(gray, mask)


def butterworth_lowpass(img: ImageArray, *params) -> ImageArray:
    """巴特沃斯低通。参数: D0, order (默认 30, 2)"""
    d0 = float(params[0]) if len(params) > 0 else 30.0
    order = int(params[1]) if len(params) > 1 else 2
    gray = to_gray(img)
    mask = _filter_mask(gray.shape, d0, "butterworth_low", order)
    return _apply_freq_filter(gray, mask)


def butterworth_highpass(img: ImageArray, *params) -> ImageArray:
    """巴特沃斯高通。参数: D0, order (默认 30, 2)"""
    d0 = float(params[0]) if len(params) > 0 else 30.0
    order = int(params[1]) if len(params) > 1 else 2
    gray = to_gray(img)
    mask = _filter_mask(gray.shape, d0, "butterworth_high", order)
    return _apply_freq_filter(gray, mask)


def gaussian_lowpass(img: ImageArray, *params) -> ImageArray:
    """高斯低通。参数: D0 (默认 30)"""
    d0 = float(params[0]) if params else 30.0
    gray = to_gray(img)
    mask = _filter_mask(gray.shape, d0, "gaussian_low")
    return _apply_freq_filter(gray, mask)


def gaussian_highpass(img: ImageArray, *params) -> ImageArray:
    """高斯高通。参数: D0 (默认 30)"""
    d0 = float(params[0]) if params else 30.0
    gray = to_gray(img)
    mask = _filter_mask(gray.shape, d0, "gaussian_high")
    return _apply_freq_filter(gray, mask)


def bandpass_filter(img: ImageArray, *params) -> ImageArray:
    """带通滤波。参数: D0 (默认 40)"""
    d0 = float(params[0]) if params else 40.0
    gray = to_gray(img)
    mask = _filter_mask(gray.shape, d0, "bandpass")
    return _apply_freq_filter(gray, mask)


def bandreject_filter(img: ImageArray, *params) -> ImageArray:
    """带阻滤波。参数: D0 (默认 40)"""
    d0 = float(params[0]) if params else 40.0
    gray = to_gray(img)
    mask = _filter_mask(gray.shape, d0, "bandreject")
    return _apply_freq_filter(gray, mask)


def fft_filter_pipeline(img: ImageArray, *params) -> ImageArray:
    """
    FFT 频域滤波完整流程演示：DFT -> 掩模 -> IDFT。
    参数: filter_type, D0 (默认 gaussian_low, 30)
    filter_type: ideal_low|ideal_high|gaussian_low|gaussian_high
    """
    ftype = str(params[0]).lower() if len(params) > 0 else "gaussian_low"
    d0 = float(params[1]) if len(params) > 1 else 30.0
    gray = to_gray(img)
    kind_map = {
        "ideal_low": "ideal_low",
        "ideal_high": "ideal_high",
        "gaussian_low": "gaussian_low",
        "gaussian_high": "gaussian_high",
        "butterworth_low": "butterworth_low",
        "butterworth_high": "butterworth_high",
    }
    kind = kind_map.get(ftype, "gaussian_low")
    mask = _filter_mask(gray.shape, d0, kind)
    return _apply_freq_filter(gray, mask)


def homomorphic_filter(img: ImageArray, *params) -> ImageArray:
    """
    同态滤波（光照校正）。
    参数: gamma_l, gamma_h, c, D0 (默认 0.5, 2.0, 1.0, 30)
    """
    gl = float(params[0]) if len(params) > 0 else 0.5
    gh = float(params[1]) if len(params) > 1 else 2.0
    c = float(params[2]) if len(params) > 2 else 1.0
    d0 = float(params[3]) if len(params) > 3 else 30.0
    gray = to_gray(img).astype(np.float64) + 1.0
    log_img = np.log(gray)
    fshift = np.fft.fftshift(np.fft.fft2(log_img))
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    u, v = np.ogrid[:rows, :cols]
    d = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
    h = (gh - gl) * (1 - np.exp(-c * (d**2) / (d0**2 + 1e-6))) + gl
    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(fshift * h)))
    out = np.exp(filtered) - 1.0
    return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
