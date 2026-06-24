# -*- coding: utf-8 -*-
"""全局教程模块注册表。"""
from __future__ import annotations
import inspect

from ch01_basics import methods as ch01
from ch02_supervised import methods as ch02
from ch03_unsupervised import methods as ch03
from ch04_dl_basics import methods as ch04
from ch05_cnn import methods as ch05
from ch06_rnn import methods as ch06
from ch07_transformer import methods as ch07
from ch08_voyo import methods as ch08


def _collect(mod):
    out = {}
    mod_name = getattr(mod, '__name__', '')
    for n, fn in inspect.getmembers(mod, inspect.isfunction):
        if n.startswith('_'):
            continue
        if getattr(fn, '__module__', None) != mod_name:
            continue
        if not (n.endswith('_demo') or n.endswith('_check') or n.endswith('_program')):
            continue
        out[n] = fn
    return out


METHOD_REGISTRY = {}
for m in (ch01, ch02, ch03, ch04, ch05, ch06, ch07, ch08):
    METHOD_REGISTRY.update(_collect(m))

ALIASES = {
    '线性回归': 'linear_regression_demo',
    '逻辑回归': 'logistic_regression_demo',
    '决策树': 'decision_tree_demo',
    'K近邻': 'knn_demo',
    '朴素贝叶斯': 'naive_bayes_demo',
    'K均值': 'kmeans_demo',
    'PCA': 'pca_demo',
    '层次聚类': 'hierarchical_demo',
    '感知机': 'perceptron_demo',
    '反向传播': 'backprop_demo',
    '激活函数': 'activation_demo',
    '损失函数': 'loss_demo',
    '卷积': 'conv2d_demo',
    '池化': 'pooling_demo',
    'CNN': 'cnn_demo',
    'RNN': 'rnn_demo',
    'LSTM': 'lstm_demo',
    '文本分类': 'text_classify_demo',
    '注意力': 'attention_demo',
    'Transformer': 'transformer_demo',
    'Voyo': 'voyo_demo',
    '环境检查': 'env_check_demo',
    '第一个程序': 'first_ml_program_demo',
    'hello': 'hello_ml_demo',
}


def resolve(name):
    k = name.strip()
    if k in METHOD_REGISTRY:
        return METHOD_REGISTRY[k]
    if k in ALIASES:
        return METHOD_REGISTRY[ALIASES[k]]
    raise KeyError('unknown: ' + name)
