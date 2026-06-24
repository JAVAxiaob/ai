# -*- coding: utf-8 -*-
"""第7章：DWT、金字塔、小波去噪/边缘/压缩。"""

from __future__ import annotations

import cv2
import numpy as np
import pywt

from utils import ImageArray, to_gray


def dwt_decompose(img: ImageArray, *params) -> ImageArray:
    """
    二维小波分解可视化（拼接 LL/LH/HL/HH）。
    参数: wavelet (默认 db1), level (默认 1)
    """
    wavelet = str(params[0]) if len(params) > 0 else "db1"
    level = int(params[1]) if len(params) > 1 else 1
    gray = to_gray(img).astype(np.float64)
    coeffs = pywt.wavedec2(gray, wavelet, level=level)
    arr, slices = pywt.coeffs_to_array(coeffs)
    return cv2.normalize(np.abs(arr), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def pyramid_downsample(img: ImageArray, *params) -> ImageArray:
    """高斯金字塔降采样。参数: levels (默认 3)"""
    levels = int(params[0]) if params else 3
    gray = to_gray(img)
    current = gray
    for _ in range(levels):
        current = cv2.pyrDown(current)
    return cv2.resize(current, (gray.shape[1], gray.shape[0]))


def wavelet_denoise(img: ImageArray, *params) -> ImageArray:
    """小波软阈值去噪。参数: wavelet, sigma (默认 db4, 自动)"""
    wavelet = str(params[0]) if len(params) > 0 else "db4"
    gray = to_gray(img).astype(np.float64)
    coeffs = pywt.wavedec2(gray, wavelet, level=2)
    detail_coeffs = coeffs[1:]
    sigma = float(params[1]) if len(params) > 1 else np.median(
        np.abs(detail_coeffs[0][0])
    ) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(gray.size))
    coeffs_thresh = [coeffs[0]]
    for detail_level in detail_coeffs:
        coeffs_thresh.append(
            tuple(pywt.threshold(c, uthresh, mode="soft") for c in detail_level)
        )
    recon = pywt.waverec2(coeffs_thresh, wavelet)
    recon = recon[: gray.shape[0], : gray.shape[1]]
    return np.clip(recon, 0, 255).astype(np.uint8)


def wavelet_edge_detect(img: ImageArray, *params) -> ImageArray:
    """小波高频子带边缘。参数: wavelet (默认 haar)"""
    wavelet = str(params[0]) if params else "haar"
    gray = to_gray(img).astype(np.float64)
    cA, (cH, cV, cD) = pywt.dwt2(gray, wavelet)
    edge = np.sqrt(cH**2 + cV**2 + cD**2)
    return cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def wavelet_compress_demo(img: ImageArray, *params) -> ImageArray:
    """小波压缩演示（丢弃高频）。参数: keep_ratio (默认 0.1)"""
    ratio = float(params[0]) if params else 0.1
    gray = to_gray(img).astype(np.float64)
    coeffs = pywt.wavedec2(gray, "db1", level=2)
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    flat = np.abs(arr.ravel())
    thresh = np.percentile(flat, (1 - ratio) * 100)
    arr_thresh = pywt.threshold(arr, thresh, mode="hard")
    coeffs2 = pywt.array_to_coeffs(arr_thresh, coeff_slices, output_format="wavedec2")
    recon = pywt.waverec2(coeffs2, "db1")
    recon = recon[: gray.shape[0], : gray.shape[1]]
    return np.clip(recon, 0, 255).astype(np.uint8)
