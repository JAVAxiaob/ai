# -*- coding: utf-8 -*-
"""第二章：监督学习。

包含：
1. 线性回归 linear_regression_demo    - 使用 NumPy 最小二乘
2. 逻辑回归 logistic_regression_demo   - 使用 sklearn
3. 决策树   decision_tree_demo         - 使用 sklearn
4. K近邻    knn_demo                   - 使用 sklearn
5. 朴素贝叶斯 naive_bayes_demo         - 使用 sklearn
"""

from __future__ import annotations

import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from utils import (
    ensure_dir, make_regression_data, make_classification_data,
    mse, accuracy, train_test_split,
    plot_regression_fit, plot_scatter_2d,
)


def linear_regression_demo(output=None):
    """最小二乘法线性回归。

    公式：theta = (X^T X)^{-1} X^T y
    """
    x, y = make_regression_data(n_samples=100, noise=5.0, seed=42)
    # 构造增广矩阵 [x, 1]
    X_aug = np.hstack([x, np.ones((len(x), 1))])
    theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
    y_pred = X_aug @ theta
    loss = mse(y, y_pred)
    print('theta (w, b) =', theta)
    print('MSE =', loss)

    out_path = None
    if output:
        out_path = os.path.join(output, 'ch02_linear_regression.png')
        ensure_dir(out_path)
    plot_regression_fit(x, y, y_pred, title='Linear Regression (OLS)', output=out_path)
    return 'mse=%.4f' % loss


def logistic_regression_demo(output=None):
    """逻辑回归：二分类。"""
    from sklearn.linear_model import LogisticRegression
    X, y = make_classification_data(n_samples=200, n_features=2, n_classes=2, seed=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=2)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print('逻辑回归准确率 =', acc)
    print('系数 =', model.coef_, '截距 =', model.intercept_)

    out_path = None
    if output:
        out_path = os.path.join(output, 'ch02_logistic.png')
        ensure_dir(out_path)
        # 绘制测试集预测
        plot_scatter_2d(X_test, y_pred, title='Logistic Regression (predictions)', output=out_path)
    return 'accuracy=%.4f' % acc


def decision_tree_demo(output=None):
    """决策树分类。"""
    from sklearn.tree import DecisionTreeClassifier
    X, y = make_classification_data(n_samples=200, n_features=2, n_classes=3, seed=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=4)
    model = DecisionTreeClassifier(max_depth=4, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print('决策树准确率 =', acc)

    out_path = None
    if output:
        out_path = os.path.join(output, 'ch02_tree.png')
        ensure_dir(out_path)
        plot_scatter_2d(X_test, y_pred, title='Decision Tree (predictions)', output=out_path)
    return 'accuracy=%.4f' % acc


def knn_demo(k=5, output=None):
    """K近邻算法。"""
    from sklearn.neighbors import KNeighborsClassifier
    X, y = make_classification_data(n_samples=200, n_features=2, n_classes=3, seed=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=6)
    model = KNeighborsClassifier(n_neighbors=int(k))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print('KNN (k=%d) 准确率 = %s' % (int(k), acc))

    out_path = None
    if output:
        out_path = os.path.join(output, 'ch02_knn.png')
        ensure_dir(out_path)
        plot_scatter_2d(X_test, y_pred, title='KNN k=%d' % int(k), output=out_path)
    return 'accuracy=%.4f' % acc


def naive_bayes_demo(output=None):
    """高斯朴素贝叶斯。"""
    from sklearn.naive_bayes import GaussianNB
    X, y = make_classification_data(n_samples=200, n_features=2, n_classes=3, seed=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=8)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print('朴素贝叶斯准确率 =', acc)

    out_path = None
    if output:
        out_path = os.path.join(output, 'ch02_nb.png')
        ensure_dir(out_path)
        plot_scatter_2d(X_test, y_pred, title='Naive Bayes (predictions)', output=out_path)
    return 'accuracy=%.4f' % acc
