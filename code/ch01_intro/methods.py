# -*- coding: utf-8 -*-
"""第1章：图像获取模拟、噪声、系统信息标注。"""

from __future__ import annotations

import cv2
import numpy as np

from utils import ImageArray, ensure_float, to_gray


def simulate_sensor_noise(img: ImageArray, *params) -> ImageArray:
    """
    模拟传感器噪声（高斯）。
    参数: sigma[, seed]
    """
    # 可变参数params第一个值为高斯分布标准差；无参数默认 15。
    # sigma 越大 → 画面噪点越重。
    sigma = float(params[0]) if params else 15.0
    seed = int(params[1]) if len(params) > 1 else None
    rng = np.random.default_rng(seed)
    # 调用工具函数把输入图像统一转为单通道灰度图，只给灰度图加噪。
    gray = to_gray(img)
    # rng.normal(均值=0, 标准差=sigma, 尺寸匹配灰度图)生成噪声矩阵；
    # 先转 float64 防止 uint8 加减溢出；
    # 原图像素叠加噪声值
    noisy = gray.astype(np.float64) + rng.normal(0, sigma, gray.shape)
    # np.clip：把小于 0、大于 255 的异常像素钳位到 0–255；
    # 转回uint8，是 OpenCV 标准图像存储类型。
    return np.clip(noisy, 0, 255).astype(np.uint8)


def simulate_quantization(img: ImageArray, *params) -> ImageArray:
    """
    模拟量化（降低灰度级）。
    参数: levels (默认 16)
    """
    levels = int(params[0]) if params else 16
    gray = to_gray(img)
    step = 256 // max(levels, 2)
    q = (gray // step) * step
    return q.astype(np.uint8)


def annotate_system_info(img: ImageArray, *params) -> ImageArray:
    """
    在图像上标注处理系统信息（模拟显示环节）。
    参数: text (可选)
    """
    text = str(params[0]) if params else "DIP System"
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    return out


def application_domains_demo(img: ImageArray, *params) -> ImageArray:
    """
    典型应用领域标注（医学/遥感/安防等）。
    参数: domain (medical|remote|security|general，默认 general)
    """
    domain = str(params[0]).lower() if params else "general"
    labels = {
        "medical": "Medical Imaging",
        "remote": "Remote Sensing",
        "security": "Security / Surveillance",
        "general": "General DIP",
    }
    text = labels.get(domain, labels["general"])
    out = img.copy()
    # 判断图像是单通道灰度图 (ndim=2)，自动转换成3 通道 BGR 彩色图，
    # 灰度图转BGR三通道，保证putText彩色文字正常渲染
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    # OpenCV绘制文字：坐标(10,30)、字体、字号0.8、橙黄色、线宽2
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    return out


def resize_for_display(img: ImageArray, *params) -> ImageArray:
    """缩放至指定宽度，模拟显示适配。参数: width (默认 512)"""
    w = int(params[0]) if params else 512
    h, wi = img.shape[:2]
    if wi <= w:
        return img
    scale = w / wi
    nh = int(h * scale)
    return cv2.resize(img, (w, nh), interpolation=cv2.INTER_AREA)
