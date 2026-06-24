# -*- coding: utf-8 -*-
"""第12章：模板匹配、分类器演示、形状匹配、CNN 特征（简化）。"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC

from utils import ImageArray, to_gray


def template_matching(img: ImageArray, *params) -> ImageArray:
    """
    模板匹配（使用图像中心块为模板）。
    参数: method_index (默认 5 = CCOEFF_NORMED)
    """
    gray = to_gray(img)
    h, w = gray.shape
    th, tw = h // 4, w // 4
    template = gray[h // 2 - th // 2 : h // 2 + th // 2, w // 2 - tw // 2 : w // 2 + tw // 2]
    method = int(params[0]) if params else cv2.TM_CCOEFF_NORMED
    res = cv2.matchTemplate(gray, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    br = (top_left[0] + tw, top_left[1] + th)
    cv2.rectangle(out, top_left, br, (0, 255, 0), 2)
    return out


def minimum_distance_classifier_demo(img: ImageArray, *params) -> ImageArray:
    """最小距离分类演示（KMeans 聚类着色）。"""
    gray = to_gray(img)
    small = cv2.resize(gray, (64, 64))
    pixels = small.reshape(-1, 1).astype(np.float64)
    k = int(params[0]) if params else 3
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(pixels)
    colored = (labels.reshape(64, 64) * (255 // k)).astype(np.uint8)
    return cv2.resize(colored, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)


def bayesian_threshold_demo(img: ImageArray, *params) -> ImageArray:
    """贝叶斯思想：两类高斯混合阈值（Otsu 近似）。"""
    gray = to_gray(img)
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def svm_segment_demo(img: ImageArray, *params) -> ImageArray:
    """SVM 像素分类演示（简化：采样训练）。"""
    gray = to_gray(img)
    h, w = gray.shape
    ys, xs = np.mgrid[0:h:8, 0:w:8]
    samples = gray[ys, xs].ravel().reshape(-1, 1).astype(np.float64)
    mid = np.median(samples)
    labels = (samples.ravel() > mid).astype(int)
    clf = SVC(kernel="linear")
    clf.fit(samples, labels)
    yy, xx = np.mgrid[0:h, 0:w]
    feat = gray.ravel().reshape(-1, 1).astype(np.float64)
    pred = clf.predict(feat).reshape(h, w)
    return (pred * 255).astype(np.uint8)


def shape_matching_hu(img: ImageArray, *params) -> ImageArray:
    """Hu 矩形状匹配标注。"""
    gray = to_gray(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hu = cv2.HuMoments(cv2.moments(cnt))
        cv2.drawContours(out, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(
            out,
            f"Hu0={hu[0][0]:.2e}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    return out


def bp_neural_demo(img: ImageArray, *params) -> ImageArray:
    """
    BP 神经网络分类演示（MLP 对像素块二分类并可视化决策区域）。
    参数: hidden_size (默认 20)
    """
    from sklearn.neural_network import MLPClassifier

    gray = to_gray(img)
    small = cv2.resize(gray, (48, 48))
    h, w = small.shape
    yy, xx = np.mgrid[0:h, 0:w]
    feats = np.column_stack([small.ravel(), xx.ravel() / w, yy.ravel() / h])
    labels = (small.ravel() > np.median(small)).astype(int)
    hidden = int(params[0]) if params else 20
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=300, random_state=0)
    clf.fit(feats, labels)
    pred = clf.predict(feats).reshape(h, w)
    return cv2.resize((pred * 255).astype(np.uint8), (gray.shape[1], gray.shape[0]))


def syntactic_pattern_demo(img: ImageArray, *params) -> ImageArray:
    """句法模式识别演示：用链码方向序列匹配简单形状（圆/方）。"""
    gray = to_gray(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if not contours:
        return out
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    shape = "circle" if len(approx) > 6 else "polygon"
    cv2.drawContours(out, [cnt], -1, (0, 255, 0), 2)
    cv2.putText(out, f"syntax:{shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def graph_matching_demo(img: ImageArray, *params) -> ImageArray:
    """
    图匹配演示：提取轮廓多边形顶点并连线匹配相似形状。
    参数: epsilon_factor (默认 0.02)
    """
    eps_f = float(params[0]) if params else 0.02
    gray = to_gray(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    graphs = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps_f * peri, True).reshape(-1, 2)
        if len(approx) >= 3:
            graphs.append(approx)
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for idx, pts in enumerate(graphs):
        color = colors[idx % len(colors)]
        for i in range(len(pts)):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % len(pts)])
            cv2.line(out, p1, p2, color, 2)
            cv2.circle(out, p1, 3, color, -1)
    if len(graphs) >= 2:
        c0 = graphs[0].mean(axis=0).astype(int)
        c1 = graphs[1].mean(axis=0).astype(int)
        cv2.line(out, tuple(c0), tuple(c1), (255, 255, 0), 1)
    return out


def cnn_feature_map_demo(img: ImageArray, *params) -> ImageArray:
    """
    深度学习特征演示：使用 OpenCV DNN 若不可用则 Sobel 堆叠模拟特征图。
    """
    gray = to_gray(img)
    try:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(gray, (224, 224)), 1.0 / 255, (224, 224), swapRB=False
        )
        # 无预训练模型时用多层滤波模拟
        raise FileNotFoundError
    except Exception:
        layers = []
        for k in (3, 5, 7):
            layers.append(cv2.GaussianBlur(gray, (k, k), 0))
        stack = np.max(np.stack(layers, axis=0), axis=0)
        return stack.astype(np.uint8)
