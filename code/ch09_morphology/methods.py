# -*- coding: utf-8 -*-
"""第9章：腐蚀膨胀、开闭、击中击不中、骨架、顶帽等。"""

from __future__ import annotations

import cv2
import numpy as np
from skimage import morphology as skimorph

from utils import ImageArray, to_gray


def _kernel(params) -> np.ndarray:
    k = int(params[0]) if params else 3
    if k % 2 == 0:
        k += 1
    return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))


def _binary(img: ImageArray) -> np.ndarray:
    gray = to_gray(img)
    _, b = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return b


def erosion(img: ImageArray, *params) -> ImageArray:
    """腐蚀。参数: kernel_size (默认 3)"""
    return cv2.erode(_binary(img), _kernel(params))


def dilation(img: ImageArray, *params) -> ImageArray:
    """膨胀。参数: kernel_size (默认 3)"""
    return cv2.dilate(_binary(img), _kernel(params))


def opening(img: ImageArray, *params) -> ImageArray:
    """开运算。参数: kernel_size (默认 3)"""
    return cv2.morphologyEx(_binary(img), cv2.MORPH_OPEN, _kernel(params))


def closing(img: ImageArray, *params) -> ImageArray:
    """闭运算。参数: kernel_size (默认 3)"""
    return cv2.morphologyEx(_binary(img), cv2.MORPH_CLOSE, _kernel(params))


def hit_or_miss(img: ImageArray, *params) -> ImageArray:
    """击中击不中变换（简化模板）。"""
    b = _binary(img)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    eroded = cv2.erode(b, kernel)
    return eroded


def boundary_extract(img: ImageArray, *params) -> ImageArray:
    """边界提取（原图减腐蚀）。"""
    b = _binary(img)
    k = _kernel(params)
    return cv2.subtract(b, cv2.erode(b, k))


def skeleton_extract(img: ImageArray, *params) -> ImageArray:
    """骨架提取。"""
    b = _binary(img) > 0
    skel = skimorph.skeletonize(b)
    return (skel.astype(np.uint8) * 255)


def thinning(img: ImageArray, *params) -> ImageArray:
    """细化（形态学骨架近似）。"""
    return skeleton_extract(img, *params)


def thickening(img: ImageArray, *params) -> ImageArray:
    """粗化（对背景腐蚀，等价于对前景膨胀的对偶操作简化）。"""
    b = _binary(img)
    k = _kernel(params)
    return cv2.dilate(b, k)


def distance_transform(img: ImageArray, *params) -> ImageArray:
    """距离变换。"""
    b = (_binary(img) > 0).astype(np.uint8)
    dist = cv2.distanceTransform(b, cv2.DIST_L2, 3)
    return cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def grayscale_erosion(img: ImageArray, *params) -> ImageArray:
    """灰度腐蚀。"""
    gray = to_gray(img)
    return cv2.erode(gray, _kernel(params))


def grayscale_dilation(img: ImageArray, *params) -> ImageArray:
    """灰度膨胀。"""
    gray = to_gray(img)
    return cv2.dilate(gray, _kernel(params))


def grayscale_opening(img: ImageArray, *params) -> ImageArray:
    """灰度开运算。"""
    gray = to_gray(img)
    return cv2.morphologyEx(gray, cv2.MORPH_OPEN, _kernel(params))


def grayscale_closing(img: ImageArray, *params) -> ImageArray:
    """灰度闭运算。"""
    gray = to_gray(img)
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, _kernel(params))


def top_hat(img: ImageArray, *params) -> ImageArray:
    """顶帽变换。"""
    gray = to_gray(img)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, _kernel(params))


def black_hat(img: ImageArray, *params) -> ImageArray:
    """黑帽变换。"""
    gray = to_gray(img)
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _kernel(params))


def morphological_gradient(img: ImageArray, *params) -> ImageArray:
    """形态学梯度。"""
    gray = to_gray(img)
    return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, _kernel(params))


def morphological_reconstruction(img: ImageArray, *params) -> ImageArray:
    """形态学重建去噪（开运算重建）。"""
    gray = to_gray(img)
    k = _kernel(params)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k)
    return cv2.subtract(gray, opened)
