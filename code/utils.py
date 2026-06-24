# -*- coding: utf-8 -*-
"""通用工具：图像读写、参数解析、类型转换。"""

from __future__ import annotations

import os
from typing import Any, List, Sequence, Tuple, Union

import cv2
import numpy as np

ImageArray = np.ndarray


def load_image(path: str) -> ImageArray:
    """加载图像，BGR 彩色或灰度。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def save_image(path: str, img: ImageArray) -> None:
    """保存图像，自动创建目录。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)


def to_gray(img: ImageArray) -> ImageArray:
    """转为单通道灰度 uint8。"""
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def ensure_float(img: ImageArray) -> np.ndarray:
    """转为 float64，范围 [0, 255]。"""
    return img.astype(np.float64)


def parse_params(param_str: Union[str, list, None]) -> List[Any]:
    """
    解析 JSON 中的参数字符串。
    例: "3, 1.5, true" -> [3, 1.5, True]
    """
    if param_str is None or param_str == "":
        return []
    if isinstance(param_str, (list, tuple)):
        return list(param_str)
    parts = [p.strip() for p in str(param_str).split(",") if p.strip()]
    result: List[Any] = []
    for p in parts:
        low = p.lower()
        if low in ("true", "yes", "1"):
            result.append(True)
        elif low in ("false", "no", "0"):
            result.append(False)
        else:
            try:
                if "." in p or "e" in low:
                    result.append(float(p))
                else:
                    result.append(int(p))
            except ValueError:
                result.append(p)
    return result


def get_kernel_size(params: Sequence[Any], default: int = 3) -> int:
    """从参数列表取奇数核尺寸。"""
    k = int(params[0]) if params else default
    k = max(3, k)
    if k % 2 == 0:
        k += 1
    return k
