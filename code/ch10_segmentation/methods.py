# -*- coding: utf-8 -*-
"""第10章：边缘、阈值、区域、分水岭分割。"""

from __future__ import annotations

import cv2
import numpy as np
from skimage import segmentation as skseg
from skimage.filters import sobel as sk_sobel

from utils import ImageArray, to_gray


def sobel_edge(img: ImageArray, *params) -> ImageArray:
    """Sobel 边缘。参数: ksize (默认 3)"""
    k = int(params[0]) if params else 3
    gray = to_gray(img)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def prewitt_edge(img: ImageArray, *params) -> ImageArray:
    """Prewitt 边缘。"""
    gray = to_gray(img).astype(np.float64)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    gx = cv2.filter2D(gray, -1, kx)
    gy = cv2.filter2D(gray, -1, ky)
    mag = np.sqrt(gx**2 + gy**2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def laplacian_edge(img: ImageArray, *params) -> ImageArray:
    """拉普拉斯边缘。参数: ksize (默认 3)"""
    k = int(params[0]) if params else 3
    gray = to_gray(img)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
    return cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def canny_edge(img: ImageArray, *params) -> ImageArray:
    """Canny 边缘。参数: low, high (默认 50, 150)"""
    low = int(params[0]) if len(params) > 0 else 50
    high = int(params[1]) if len(params) > 1 else 150
    return cv2.Canny(to_gray(img), low, high)


def threshold_global(img: ImageArray, *params) -> ImageArray:
    """全局固定阈值分割。参数: thresh[, maxval] (默认 127, 255)"""
    t = int(params[0]) if len(params) > 0 else 127
    mv = int(params[1]) if len(params) > 1 else 255
    _, out = cv2.threshold(to_gray(img), t, mv, cv2.THRESH_BINARY)
    return out


def otsu_threshold(img: ImageArray, *params) -> ImageArray:
    """Otsu 阈值分割。"""
    gray = to_gray(img)
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def adaptive_threshold(img: ImageArray, *params) -> ImageArray:
    """自适应阈值。参数: block_size, C (默认 11, 2)"""
    bs = int(params[0]) if len(params) > 0 else 11
    c = int(params[1]) if len(params) > 1 else 2
    if bs % 2 == 0:
        bs += 1
    gray = to_gray(img)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, c
    )


def multi_threshold(img: ImageArray, *params) -> ImageArray:
    """多阈值（双阈值三类）。参数: t1, t2 (默认 85, 170)"""
    t1 = int(params[0]) if len(params) > 0 else 85
    t2 = int(params[1]) if len(params) > 1 else 170
    gray = to_gray(img)
    out = np.zeros_like(gray)
    out[(gray >= t1) & (gray < t2)] = 128
    out[gray >= t2] = 255
    return out


def region_growing(img: ImageArray, *params) -> ImageArray:
    """区域生长。参数: row, col, threshold (默认中心, 15)"""
    gray = to_gray(img)
    h, w = gray.shape
    sr = int(params[0]) if len(params) > 0 else h // 2
    sc = int(params[1]) if len(params) > 1 else w // 2
    th = int(params[2]) if len(params) > 2 else 15
    seed_val = int(gray[sr, sc])
    mask = np.zeros((h, w), np.uint8)
    stack = [(sr, sc)]
    mask[sr, sc] = 255
    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] == 0:
                if abs(int(gray[nr, nc]) - seed_val) <= th:
                    mask[nr, nc] = 255
                    stack.append((nr, nc))
    return mask


def region_split_merge(img: ImageArray, *params) -> ImageArray:
    """区域分裂合并（四叉树简化：均匀块合并）。"""
    gray = to_gray(img).astype(np.float64)
    h, w = gray.shape
    block = int(params[0]) if params else 32
    out = gray.copy()
    for i in range(0, h, block):
        for j in range(0, w, block):
            patch = gray[i : i + block, j : j + block]
            out[i : i + block, j : j + block] = patch.mean()
    return out.astype(np.uint8)


def watershed_segment(img: ImageArray, *params) -> ImageArray:
    """分水岭分割。参数: markers_count (默认 3)"""
    gray = to_gray(img)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(thresh, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    if img.ndim == 3:
        color = img
    else:
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    out = np.zeros_like(gray)
    out[markers > 1] = 255
    return out


def combined_edge_region(img: ImageArray, *params) -> ImageArray:
    """边缘与区域结合：Canny 边缘与 Otsu 区域相乘。"""
    edges = canny_edge(img, 50, 150)
    region = otsu_threshold(img)
    return cv2.bitwise_and(edges, region)
