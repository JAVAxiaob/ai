# -*- coding: utf-8 -*-
"""第5章：噪声、退化、维纳/逆滤波、投影重建演示。"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift

from utils import ImageArray, get_kernel_size, to_gray


def add_gaussian_noise(img: ImageArray, *params) -> ImageArray:
    """添加高斯噪声。参数: sigma (默认 25)"""
    sigma = float(params[0]) if params else 25.0
    g = to_gray(img).astype(np.float64)
    noise = np.random.default_rng(0).normal(0, sigma, g.shape)
    return np.clip(g + noise, 0, 255).astype(np.uint8)


def add_periodic_noise(img: ImageArray, *params) -> ImageArray:
    """添加周期噪声（正弦条纹）。参数: amplitude, frequency_x, frequency_y (默认 30, 0.15, 0.1)"""
    amp = float(params[0]) if len(params) > 0 else 30.0
    fx = float(params[1]) if len(params) > 1 else 0.15
    fy = float(params[2]) if len(params) > 2 else 0.1
    gray = to_gray(img).astype(np.float64)
    h, w = gray.shape
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    pattern = amp * np.sin(2 * np.pi * fx * xx / w) * np.sin(2 * np.pi * fy * yy / h)
    return np.clip(gray + pattern, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img: ImageArray, *params) -> ImageArray:
    """椒盐噪声。参数: amount (默认 0.05)"""
    amount = float(params[0]) if params else 0.05
    out = to_gray(img).copy()
    rng = np.random.default_rng(0)
    n = int(amount * out.size)
    coords = (rng.integers(0, out.shape[0], n), rng.integers(0, out.shape[1], n))
    out[coords] = 255
    coords = (rng.integers(0, out.shape[0], n), rng.integers(0, out.shape[1], n))
    out[coords] = 0
    return out


def motion_blur(img: ImageArray, *params) -> ImageArray:
    """运动模糊。参数: size, angle (默认 15, 0)"""
    size = int(params[0]) if len(params) > 0 else 15
    angle = float(params[1]) if len(params) > 1 else 0.0
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0 / size
    m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, m, (size, size))
    gray = to_gray(img)
    return cv2.filter2D(gray, -1, kernel)


def defocus_blur(img: ImageArray, *params) -> ImageArray:
    """散焦模糊（圆盘核）。参数: radius (默认 5)"""
    r = int(params[0]) if params else 5
    k = 2 * r + 1
    kernel = np.zeros((k, k), np.float64)
    cv2.circle(kernel, (r, r), r, 1.0, -1)
    kernel /= kernel.sum() + 1e-6
    return cv2.filter2D(to_gray(img), -1, kernel)


def mean_denoise(img: ImageArray, *params) -> ImageArray:
    """均值去噪。参数: kernel_size (默认 5)"""
    k = get_kernel_size(params, 5)
    return cv2.blur(to_gray(img), (k, k))


def median_denoise(img: ImageArray, *params) -> ImageArray:
    """中值去噪。参数: kernel_size (默认 5)"""
    k = get_kernel_size(params, 5)
    return cv2.medianBlur(to_gray(img), k)


def adaptive_denoise(img: ImageArray, *params) -> ImageArray:
    """自适应滤波（双边）。参数: d, sigma_color, sigma_space"""
    d = int(params[0]) if len(params) > 0 else 9
    sc = float(params[1]) if len(params) > 1 else 75.0
    ss = float(params[2]) if len(params) > 2 else 75.0
    return cv2.bilateralFilter(to_gray(img), d, sc, ss)


def wiener_filter_spatial(img: ImageArray, *params) -> ImageArray:
    """维纳滤波（频域实现）。参数: K (默认 0.01)"""
    k = float(params[0]) if params else 0.01
    gray = to_gray(img).astype(np.float64)
    f = fftshift(fft2(gray))
    psf = np.ones((3, 3)) / 9.0
    h = np.zeros_like(gray)
    h[:3, :3] = psf
    h = fftshift(fft2(h))
    wiener = np.conj(h) / (np.abs(h) ** 2 + k) * f
    out = np.real(ifft2(ifftshift(wiener)))
    return np.clip(out, 0, 255).astype(np.uint8)


def inverse_filter(img: ImageArray, *params) -> ImageArray:
    """逆滤波（频域，带阈值防除零）。参数: threshold (默认 0.01)"""
    th = float(params[0]) if params else 0.01
    gray = to_gray(img).astype(np.float64)
    f = fftshift(fft2(gray))
    psf = np.ones((5, 5)) / 25.0
    h = np.zeros_like(gray)
    h[:5, :5] = psf
    h = fftshift(fft2(h))
    h_mag = np.abs(h)
    h_inv = np.conj(h) / (h_mag**2 + th)
    out = np.real(ifft2(ifftshift(f * h_inv)))
    return np.clip(out, 0, 255).astype(np.uint8)


def wiener_filter_freq(img: ImageArray, *params) -> ImageArray:
    """频域维纳滤波。参数: K (默认 0.01)"""
    return wiener_filter_spatial(img, *params)


def constrained_least_squares(img: ImageArray, *params) -> ImageArray:
    """约束最小二乘滤波近似（拉普拉斯正则）。参数: gamma (默认 0.1)"""
    gamma = float(params[0]) if params else 0.1
    gray = to_gray(img).astype(np.float64)
    lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    p = np.zeros_like(gray)
    p[:3, :3] = lap
    f = fftshift(fft2(gray))
    h = np.ones((5, 5)) / 25.0
    psf = np.zeros_like(gray)
    psf[:5, :5] = h
    h_f = fftshift(fft2(psf))
    p_f = fftshift(fft2(p))
    denom = np.abs(h_f) ** 2 + gamma * np.abs(p_f) ** 2 + 1e-6
    out = np.real(ifft2(ifftshift(np.conj(h_f) / denom * f)))
    return np.clip(out, 0, 255).astype(np.uint8)


def blind_deconvolution(img: ImageArray, *params) -> ImageArray:
    """盲复原简化（Richardson-Lucy 近似，迭代少）。参数: iterations (默认 5)"""
    iters = int(params[0]) if params else 5
    from skimage import restoration

    gray = to_gray(img).astype(np.float64) / 255.0
    psf = np.ones((5, 5)) / 25.0
    result, _ = restoration.unsupervised_wiener(gray, psf)
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def radon_reconstruction_demo(img: ImageArray, *params) -> ImageArray:
    """
    Radon 变换与反投影重建演示。
    参数: num_angles (默认 180)
    """
    from skimage.transform import iradon, radon

    gray = to_gray(img)
    size = min(gray.shape)
    gray = cv2.resize(gray, (size, size))
    theta = np.linspace(0.0, 180.0, int(params[0]) if params else 180, endpoint=False)
    sinogram = radon(gray, theta=theta, circle=True)
    recon = iradon(sinogram, theta=theta, circle=True)
    return cv2.normalize(recon, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
