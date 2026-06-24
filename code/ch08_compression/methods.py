# -*- coding: utf-8 -*-
"""第8章：霍夫曼、行程、LZW、DCT/JPEG 演示。"""

from __future__ import annotations

import io
import zlib
from collections import Counter
from heapq import heappop, heappush

import cv2
import numpy as np
from scipy.fftpack import dct, idct

from utils import ImageArray, to_gray


class _Node:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def _build_huffman(data: bytes) -> dict:
    freq = Counter(data)
    heap = []
    for sym, f in freq.items():
        heappush(heap, _Node(f, sym))
    if not heap:
        return {}
    while len(heap) > 1:
        a, b = heappop(heap), heappop(heap)
        heappush(heap, _Node(a.freq + b.freq, left=a, right=b))
    root = heap[0]
    codes = {}

    def walk(node, code=""):
        if node.symbol is not None:
            codes[node.symbol] = code or "0"
            return
        walk(node.left, code + "0")
        walk(node.right, code + "1")

    walk(root)
    return codes


def arithmetic_encode_vis(img: ImageArray, *params) -> ImageArray:
    """算术编码压缩比演示（使用 zlib 作为工程实现参考）。"""
    gray = to_gray(img)
    raw = gray.tobytes()
    comp = zlib.compress(raw, level=9)
    ratio = len(raw) / max(len(comp), 1)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        out,
        f"Arithmetic/zlib ratio~{ratio:.2f}x",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )
    return out


def video_frame_prediction_demo(img: ImageArray, *params) -> ImageArray:
    """
    视频帧间预测演示：用平移模拟第二帧，显示帧间差分（运动估计简化）。
    参数: shift_x, shift_y (默认 5, 3)
    """
    sx = int(params[0]) if len(params) > 0 else 5
    sy = int(params[1]) if len(params) > 1 else 3
    frame1 = _ensure_bgr_for_video(img)
    m = np.float32([[1, 0, sx], [0, 1, sy]])
    frame2 = cv2.warpAffine(frame1, m, (frame1.shape[1], frame1.shape[0]))
    diff = cv2.absdiff(frame1, frame2)
    return cv2.cvtColor(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def _ensure_bgr_for_video(img: ImageArray) -> ImageArray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def huffman_encode_decode_vis(img: ImageArray, *params) -> ImageArray:
    """霍夫曼编码演示（在图像上标注压缩比）。"""
    gray = to_gray(img)
    data = gray.tobytes()
    codes = _build_huffman(data)
    encoded_len = sum(len(codes[b]) for b in data)
    ratio = len(data) * 8 / max(encoded_len, 1)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        out,
        f"Huffman ratio~{ratio:.2f}x",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return out


def run_length_encode_vis(img: ImageArray, *params) -> ImageArray:
    """行程编码可视化（二值图）。"""
    gray = to_gray(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    flat = binary.flatten()
    runs = []
    prev, count = flat[0], 1
    for v in flat[1:]:
        if v == prev:
            count += 1
        else:
            runs.append((int(prev), count))
            prev, count = v, 1
    runs.append((int(prev), count))
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        out,
        f"RLE runs={len(runs)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return out


def lzw_compress_vis(img: ImageArray, *params) -> ImageArray:
    """LZW（zlib）压缩比演示。"""
    gray = to_gray(img)
    raw = gray.tobytes()
    comp = zlib.compress(raw)
    ratio = len(raw) / max(len(comp), 1)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        out,
        f"zlib ratio~{ratio:.2f}x",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return out


def dct_compress(img: ImageArray, *params) -> ImageArray:
    """DCT 块压缩。参数: keep_coeffs (默认 16，8x8 块保留系数个数)"""
    keep = int(params[0]) if params else 16
    gray = to_gray(img).astype(np.float64) - 128.0
    h, w = gray.shape
    bh, bw = (h // 8) * 8, (w // 8) * 8
    gray = gray[:bh, :bw]
    out = np.zeros_like(gray)
    for i in range(0, bh, 8):
        for j in range(0, bw, 8):
            block = gray[i : i + 8, j : j + 8]
            d = dct(dct(block.T, norm="ortho").T, norm="ortho")
            flat = np.abs(d.flatten())
            thresh = np.sort(flat)[-keep] if keep < 64 else 0
            d[np.abs(d) < thresh] = 0
            block2 = idct(idct(d.T, norm="ortho").T, norm="ortho")
            out[i : i + 8, j : j + 8] = block2
    out = out + 128.0
    if bh < h or bw < w:
        full = to_gray(img).astype(np.float64)
        full[:bh, :bw] = out
        out = full
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg_quality_compress(img: ImageArray, *params) -> ImageArray:
    """JPEG 有损压缩。参数: quality (默认 30)"""
    q = int(params[0]) if params else 30
    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return img
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return decoded


def jpeg2000_compress(img: ImageArray, *params) -> ImageArray:
    """JPEG2000 压缩（需 OpenCV 支持）。参数: quality (默认 20)"""
    q = int(params[0]) if params else 20
    gray = to_gray(img)
    try:
        ok, buf = cv2.imencode(
            ".jp2", gray, [cv2.IMWRITE_JPEG2000_QUALITY, q]
        )
        if ok:
            return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except Exception:
        pass
    return jpeg_quality_compress(img, q)
