# -*- coding: utf-8 -*-
"""第一章：机器学习基础。"""

from __future__ import annotations

import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from utils import ensure_dir, make_regression_data, mse, plot_regression_fit


def hello_ml_demo(*args, **kwargs):
    """打印欢迎信息，展示机器学习概念。"""
    msg = """
    ==================================================
    欢迎来到机器学习教程！
    ==================================================
    机器学习（Machine Learning, ML）是人工智能的一个分支，
    它让计算机从 数据 中学习规律（模型），然后用这个规律
    对新数据进行预测或决策。

    三类主要学习方式：
      - 监督学习（Supervised Learning）：有标签，如分类、回归
      - 无监督学习（Unsupervised Learning）：无标签，如聚类、降维
      - 强化学习（Reinforcement Learning）：通过奖励信号学习

    本教程使用 NumPy + scikit-learn，逐步实现经典算法。
    ==================================================
    """
    print(msg)
    return 'welcome'


def env_check_demo(*args, **kwargs):
    """检查当前 Python 环境及主要依赖版本。"""
    import platform
    print('Python :', platform.python_version())
    print('NumPy  :', np.__version__)
    try:
        import sklearn
        print('sklearn:', sklearn.__version__)
    except Exception as e:
        print('sklearn: 未安装 -', e)
    try:
        import matplotlib
        print('matplotlib:', matplotlib.__version__)
    except Exception as e:
        print('matplotlib: 未安装 -', e)
    return 'env-ok'


def first_ml_program_demo(epochs=200, lr=0.01, output=None):
    """第一个机器学习程序：手动实现线性拟合。

    模型：y_hat = w * x + b
    损失：MSE = mean((y_hat - y) ** 2)
    优化：梯度下降
    """
    x, y = make_regression_data(n_samples=80, noise=4.0, seed=0)
    x_flat = x.ravel()
    w = 0.0
    b = 0.0
    n = len(y)
    for ep in range(1, int(epochs) + 1):
        y_pred = w * x_flat + b
        err = y_pred - y
        grad_w = (2.0 / n) * np.sum(err * x_flat)
        grad_b = (2.0 / n) * np.sum(err)
        w -= float(lr) * grad_w
        b -= float(lr) * grad_b
        if ep % 40 == 0:
            loss = mse(y, y_pred)
            print('epoch %d: w=%.4f, b=%.4f, mse=%.4f' % (ep, w, b, loss))

    y_pred = w * x_flat + b
    print('最终参数: w=%.4f, b=%.4f' % (w, b))
    print('真实参数: w=2.5, b=3.0 (数据生成时使用)')

    out_path = None
    if output:
        out_path = os.path.join(output, 'ch01_first_ml.png')
        ensure_dir(out_path)
    plot_regression_fit(x, y, y_pred, title='First ML: Linear Fit', output=out_path)
    return 'w=%.4f,b=%.4f' % (w, b)
