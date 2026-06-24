# -*- coding: utf-8 -*-
"""第五章：卷积神经网络 CNN。

包含：
1. 2D 卷积演示 conv2d_demo  - NumPy 手动实现并对示例图像卷积
2. 池化演示    pooling_demo - 手动实现最大/平均池化
3. 小型 CNN    cnn_demo     - 使用 sklearn（简易图像特征 + 分类器）
"""

from __future__ import annotations

import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from utils import ensure_dir


def _make_demo_image(size=64):
    """生成一个带边缘和十字的演示图像。"""
    img = np.zeros((size, size), dtype=float)
    img[10:30, 10:30] = 1.0
    img[35:55, 35:55] = 0.8
    img[20:45, 30] = 0.6
    img[30, 20:45] = 0.6
    return img


def _conv2d(img, kernel):
    """对 2D 图像做 valid 卷积。"""
    kh, kw = kernel.shape
    h, w = img.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * kernel)
    return out


def _pool(img, k=2, kind='max'):
    """k x k 池化。"""
    h, w = img.shape
    nh, nw = h // k, w // k
    out = np.zeros((nh, nw))
    for i in range(nh):
        for j in range(nw):
            patch = img[i * k:(i + 1) * k, j * k:(j + 1) * k]
            out[i, j] = patch.max() if kind == 'max' else patch.mean()
    return out


def conv2d_demo(output=None):
    """对示例图像做 Sobel/均值/锐化卷积核演示。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img = _make_demo_image(64)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = sobel_x.T
    mean_k = np.ones((3, 3)) / 9.0
    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)

    results = {
        'original': img,
        'sobel_x': _conv2d(img, sobel_x),
        'sobel_y': _conv2d(img, sobel_y),
        'mean': _conv2d(img, mean_k),
        'sharpen': _conv2d(img, sharpen),
    }
    for name, mat in results.items():
        print('conv2d %s shape=' % name, mat.shape, 'min=%.2f max=%.2f' % (mat.min(), mat.max()))

    if output:
        ensure_dir(output)
        fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
        for ax, (name, mat) in zip(axes.ravel(), results.items()):
            ax.imshow(mat, cmap='gray')
            ax.set_title(name)
            ax.axis('off')
        fp = os.path.join(output, 'ch05_conv2d.png')
        plt.savefig(fp, dpi=100, bbox_inches='tight')
        print('saved:', fp)
        plt.close()
    return 'conv2d-done'


def pooling_demo(output=None):
    """最大池化与平均池化演示。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img = _make_demo_image(64)
    mp = _pool(img, k=2, kind='max')
    ap = _pool(img, k=2, kind='mean')
    print('原图形状:', img.shape, '最大池化:', mp.shape, '平均池化:', ap.shape)

    if output:
        ensure_dir(output)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img, cmap='gray'); axes[0].set_title('Original')
        axes[1].imshow(mp, cmap='gray'); axes[1].set_title('MaxPool 2x2')
        axes[2].imshow(ap, cmap='gray'); axes[2].set_title('AvgPool 2x2')
        for ax in axes:
            ax.axis('off')
        fp = os.path.join(output, 'ch05_pooling.png')
        plt.savefig(fp, dpi=100, bbox_inches='tight')
        print('saved:', fp)
        plt.close()
    return 'pooling-done'


def cnn_demo(epochs=30, output=None):
    """小型 "CNN" 演示：使用手动卷积特征 + MLP 分类。"""
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    digits = load_digits()
    X = digits.images  # (n, 8, 8)
    y = digits.target

    # 用一个 Sobel 卷积核提取边缘特征
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    feats = []
    for img in X:
        c1 = _conv2d(img, sobel)
        c2 = _conv2d(img, sobel.T)
        # 手动最大池化 + 展平
        f1 = _pool(c1, k=2, kind='max').ravel()
        f2 = _pool(c2, k=2, kind='max').ravel()
        feats.append(np.concatenate([img.ravel(), f1, f2]))
    X_feat = np.array(feats)

    X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.3, random_state=0)
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=int(epochs),
                          random_state=0, learning_rate_init=0.001)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print('小型 CNN 演示准确率 =', acc)

    if output:
        ensure_dir(output)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(model.loss_curve_)
        plt.title('CNN-demo training loss')
        plt.xlabel('epoch'); plt.ylabel('loss')
        plt.grid(True, alpha=0.3)
        fp = os.path.join(output, 'ch05_cnn_loss.png')
        plt.savefig(fp, dpi=100, bbox_inches='tight')
        print('saved:', fp)
        plt.close()
    return 'accuracy=%.4f' % acc
