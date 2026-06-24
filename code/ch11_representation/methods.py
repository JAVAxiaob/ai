# -*- coding: utf-8 -*-
"""第11章：边界、区域描述、PCA、Harris/SIFT/ORB。"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import label, regionprops

from utils import ImageArray, to_gray


def marked_graph_boundary(img: ImageArray, *params) -> ImageArray:
    """标记图边界：为每个连通区域分配不同颜色边界。"""
    gray = to_gray(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    num, labels = cv2.connectedComponents(binary)
    out = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
    for lab in range(1, num):
        mask = (labels == lab).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (
            int(37 * lab % 255),
            int(17 * lab % 255),
            int(97 * lab % 255),
        )
        cv2.drawContours(out, contours, -1, color, 1)
    return out


def chain_code_boundary(img: ImageArray, *params) -> ImageArray:
    """链码边界绘制。"""
    gray = to_gray(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if contours:
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, (0, 255, 0), 1)
    return out


def polygon_approximation(img: ImageArray, *params) -> ImageArray:
    """多边形近似。参数: epsilon_factor (默认 0.02)"""
    eps_f = float(params[0]) if params else 0.02
    gray = to_gray(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps_f * peri, True)
        cv2.drawContours(out, [approx], -1, (0, 0, 255), 2)
    return out


def region_moments_vis(img: ImageArray, *params) -> ImageArray:
    """区域矩与几何特征可视化（面积、周长、圆形度、偏心率）。"""
    gray = to_gray(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    labeled = label(binary > 0)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for prop in regionprops(labeled):
        y, x = prop.centroid
        circ = 4 * np.pi * prop.area / (prop.perimeter**2 + 1e-6)
        ecc = prop.eccentricity
        cv2.circle(out, (int(x), int(y)), 4, (255, 0, 0), -1)
        text = f"A={prop.area:.0f} C={circ:.2f} e={ecc:.2f}"
        cv2.putText(out, text, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    return out


def texture_spectral_vis(img: ImageArray, *params) -> ImageArray:
    """纹理频谱特征：对数幅度谱可视化。"""
    gray = to_gray(img).astype(np.float64)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def texture_structural_vis(img: ImageArray, *params) -> ImageArray:
    """纹理结构特征：局部梯度方向一致性。"""
    gray = to_gray(img).astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    angle = np.arctan2(gy, gx)
    k = int(params[0]) if params else 7
    if k % 2 == 0:
        k += 1
    mean_sin = cv2.blur(np.sin(angle), (k, k))
    mean_cos = cv2.blur(np.cos(angle), (k, k))
    coherence = np.sqrt(mean_sin**2 + mean_cos**2)
    return (coherence * 255).astype(np.uint8)


def texture_glcm_vis(img: ImageArray, *params) -> ImageArray:
    """纹理统计（局部方差图）。"""
    gray = to_gray(img).astype(np.float64)
    k = int(params[0]) if params else 9
    if k % 2 == 0:
        k += 1
    mean = cv2.blur(gray, (k, k))
    sq_mean = cv2.blur(gray**2, (k, k))
    var = sq_mean - mean**2
    return cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def pca_projection(img: ImageArray, *params) -> ImageArray:
    """PCA 降维重建。参数: components (默认 50)"""
    n_comp = int(params[0]) if params else 50
    gray = to_gray(img).astype(np.float64)
    h, w = gray.shape
    data = gray.reshape(-1, w)
    mean = data.mean(axis=0)
    centered = data - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:n_comp]
    basis = eigvecs[:, idx]
    proj = centered @ basis @ basis.T
    recon = proj + mean
    return np.clip(recon, 0, 255).astype(np.uint8)


def harris_corners(img: ImageArray, *params) -> ImageArray:
    """Harris 角点。参数: block_size, k, threshold (默认 2, 0.04, 0.01)"""
    bs = int(params[0]) if len(params) > 0 else 2
    k = float(params[1]) if len(params) > 1 else 0.04
    th = float(params[2]) if len(params) > 2 else 0.01
    gray = np.float32(to_gray(img))
    dst = cv2.cornerHarris(gray, bs, 3, k)
    out = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    out[dst > th * dst.max()] = [0, 0, 255]
    return out


def sift_features(img: ImageArray, *params) -> ImageArray:
    """SIFT 特征点绘制。"""
    gray = to_gray(img)
    sift = cv2.SIFT_create()
    kp, _ = sift.detectAndCompute(gray, None)
    return cv2.drawKeypoints(
        gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )


def surf_features(img: ImageArray, *params) -> ImageArray:
    """SURF/ORB 替代（OpenCV 4 可能无 SURF）。使用 ORB 若 SURF 不可用。"""
    gray = to_gray(img)
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
        kp, _ = surf.detectAndCompute(gray, None)
    except Exception:
        orb = cv2.ORB_create(500)
        kp, _ = orb.detectAndCompute(gray, None)
    return cv2.drawKeypoints(gray, kp, None, color=(0, 255, 0))


def orb_features(img: ImageArray, *params) -> ImageArray:
    """ORB 特征点。参数: nfeatures (默认 500)"""
    n = int(params[0]) if params else 500
    gray = to_gray(img)
    orb = cv2.ORB_create(n)
    kp, _ = orb.detectAndCompute(gray, None)
    return cv2.drawKeypoints(gray, kp, None, color=(255, 0, 0))
