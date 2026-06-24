# -*- coding: utf-8 -*-
"""第六章：循环神经网络 RNN / LSTM。

包含：
1. vanilla RNN 前向传播  rnn_demo       - NumPy 手动实现
2. LSTM 单元           lstm_demo        - NumPy 手动实现前向传播
3. 文本分类（词袋模型 text_classify_demo - 使用 sklearn
"""

from __future__ import annotations

import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from utils import ensure_dir


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _tanh(x):
    return np.tanh(x)


def rnn_demo(seq_len=10, input_dim=4, hidden=8, output=None):
    """Elman RNN 前向传播：h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)。"""
    rng = np.random.default_rng(0)
    W_xh = rng.normal(0, 0.1, (input_dim, hidden))
    W_hh = rng.normal(0, 0.1, (hidden, hidden))
    b_h = np.zeros(hidden)
    h = np.zeros(hidden)
    xs = rng.normal(0, 0.5, (seq_len, input_dim))
    hs = []
    for t in range(seq_len):
        h = _tanh(xs[t] @ W_xh + h @ W_hh + b_h)
        hs.append(h.copy())
    print('RNN: 序列长度=%d, 隐藏状态形状=%s' % (seq_len, str(h.shape)))

    if output:
        ensure_dir(output)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.imshow(np.array(hs).T, aspect='auto', cmap='viridis')
        plt.title('RNN hidden states over time')
        plt.xlabel('time step'); plt.ylabel('hidden dim')
        fp = os.path.join(output, 'ch06_rnn.png')
        plt.savefig(fp, dpi=100, bbox_inches='tight')
        print('saved:', fp)
        plt.close()
    return 'rnn-done'


def lstm_demo(seq_len=10, input_dim=4, hidden=8, output=None):
    """LSTM 单元前向传播。"""
    rng = np.random.default_rng(1)
    # 四个门共用权重：遗忘、输入、候选、输出
    sz = 4 * hidden
    Wx = rng.normal(0, 0.1, (input_dim, sz))
    Wh = rng.normal(0, 0.1, (hidden, sz))
    b = np.zeros(sz)
    h = np.zeros(hidden)
    c = np.zeros(hidden)
    xs = rng.normal(0, 0.5, (seq_len, input_dim))
    hs = []
    for t in range(seq_len):
        gates = xs[t] @ Wx + h @ Wh + b
        i, f, g, o = np.split(gates, 4)
        i = _sigmoid(i); f = _sigmoid(f); g = _tanh(g); o = _sigmoid(o)
        c = f * c + i * g
        h = o * _tanh(c)
        hs.append(h.copy())
    print('LSTM: 序列长度=%d, 隐藏形状=%s' % (seq_len, str(h.shape)))

    if output:
        ensure_dir(output)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.imshow(np.array(hs).T, aspect='auto', cmap='viridis')
        plt.title('LSTM hidden states over time')
        plt.xlabel('time step'); plt.ylabel('hidden dim')
        fp = os.path.join(output, 'ch06_lstm.png')
        plt.savefig(fp, dpi=100, bbox_inches='tight')
        print('saved:', fp)
        plt.close()
    return 'lstm-done'


def text_classify_demo(output=None):
    """简易文本分类：使用 sklearn 的 20 新闻组（如果不可用则用自带小语料）。"""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score

    # 构造一个小的二分类语料
    texts = [
        '机器学习 算法 深度 学习 神经网络 训练 模型',
        '深度 学习 卷积 网络 图像 识别 CNN',
        '神经网络 训练 梯度 下降',
        '机器 学习 分类 回归 算法',
        '足球 比赛 球员 进球 体育',
        '篮球 比赛 球员 得分 体育',
        '体育 足球 篮球 运动',
        '比赛 得分 体育 运动员',
    ] * 4
    labels = [0, 0, 0, 0, 1, 1, 1, 1] * 4

    test_texts = [
        '深度 学习 神经网络 训练',
        '卷积 网络 图像',
        '足球 比赛 体育',
        '篮球 运动员 得分',
    ]
    test_labels = [0, 0, 1, 1]

    vec = CountVectorizer()
    X = vec.fit_transform(texts).toarray()
    X_test = vec.transform(test_texts).toarray()

    model = MultinomialNB()
    model.fit(X, labels)
    preds = model.predict(X_test)
    acc = accuracy_score(test_labels, preds)
    print('文本分类准确率 =', acc)
    print('预测值:', preds, '真实值:', test_labels)
    return 'accuracy=%.4f' % acc
