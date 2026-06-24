# -*- coding: utf-8 -*-
"""生成测试图像到 img 目录。"""

import os

import cv2
import numpy as np

IMG_DIR = r"D:\job_files\图像算法\img"


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    h, w = 480, 640
    # 测试图1：渐变+几何
    img1 = np.zeros((h, w, 3), np.uint8)
    for i in range(h):
        img1[i, :, 0] = i * 255 // h
        img1[i, :, 1] = 128
    cv2.rectangle(img1, (100, 100), (400, 350), (0, 255, 255), -1)
    cv2.circle(img1, (500, 200), 80, (255, 0, 0), -1)
    cv2.putText(img1, "Test1", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(IMG_DIR, "test1.png"), img1)

    # 测试图2：噪声+纹理
    rng = np.random.default_rng(42)
    base = rng.integers(80, 180, (h, w), dtype=np.uint8)
    img2 = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    for k in range(5):
        cx, cy = rng.integers(50, w - 50), rng.integers(50, h - 50)
        c = int(rng.integers(0, 255))
        cv2.circle(img2, (cx, cy), int(rng.integers(20, 60)), (c, c, c), -1)
    noise = rng.normal(0, 20, base.shape)
    gray = np.clip(base.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    img2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(IMG_DIR, "test2.png"), img2)
    print(f"已生成: {IMG_DIR}/test1.png, test2.png")


if __name__ == "__main__":
    main()
