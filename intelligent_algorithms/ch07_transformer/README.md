# 第七章：Transformer 与注意力机制

> 面向对象：零基础小白
> 本章目标：掌握「缩放点积注意力」，并理解 Transformer 的前向传播。

---

## 一、章节知识点

### 7.1 注意力机制

**直觉**：一句话里预测下一个词时，你需要「看」前面哪些词更重要 ——
注意力机制就是给每个词一个「关注度权重」，加权求和得到当前表示。

**缩放点积注意力（Scaled Dot-Product Attention）**：
`
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V
`
- Q = query（当前「发出提问」的位置）
- K = key（被「查阅」的所有位置）
- V = value（对应位置的信息）
- 除以 sqrt(d_k) 是为了让内积在高维时不至于过大，
  否则 softmax 的梯度会趋近 0。

**多头注意力（Multi-Head Attention）**：
把 Q/K/V 先投影到 h 个子空间，分别做注意力，再拼接。
这样每个头可以关注不同的「关系类型」（语法/语义/指代等）。

对应代码：ttention_demo() 画一个 seq_len × seq_len 的注意力权重热力图。

### 7.2 Transformer 架构

**核心论文**：*Attention Is All You Need* (Vaswani et al., 2017)

一个 Transformer Block = 多头注意力 + 前馈网络 (FFN) + 两次残差 + 两次层归一化：

`
x → LayerNorm → MultiHeadSelfAttention → Dropout → +x（残差）
  → LayerNorm → Linear → GELU → Linear → Dropout → +x（残差）
`

FFN 通常是 Linear(d_model → 4*d_model) → GELU → Linear(4*d_model → d_model)。

**位置编码（Positional Encoding）**：
注意力本身是「位置无关」的（只是加权平均），
必须显式把位置信息注入。原始论文用正弦/余弦：
`
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
`
现在的主流模型（BERT、GPT 系列）改用**可学习的位置嵌入**。

对应代码：	ransformer_demo(d_model=16, num_heads=4, seq_len=8)
- 单块 Transformer 前向；
- 输出形状 (seq_len, d_model)；
- 热力图保存在 output/ch07_transformer.png。

---

## 二、横向对比

### RNN vs Transformer

| 维度 | RNN/LSTM | Transformer |
|------|---------|------------|
| 并行性 | ❌ 必须按时间顺序处理 | ✅ 整个序列一次计算 |
| 长距离依赖 | ⭐⭐ 靠门控仍有衰减 | ✅ O(1) 跳跃直达 |
| 内存占用 | O(T) 序列长度 | O(T^2 * d) 注意力矩阵 |
| 小样本表现 | ⭐⭐⭐⭐ | ⭐⭐ （在非常小数据上可能被 LSTM 反超） |
| 归纳偏置 | 时间顺序强 | 无（靠位置编码注入） |
| 典型代表 | LSTM, GRU | BERT, GPT, T5, ViT |

### 不同注意力变体

| 注意力 | 时间复杂度 | 用途 |
|--------|----------|------|
| 全连接自注意力 | O(T^2 * d) | 默认，T<1024 可行 |
| 稀疏注意力 | O(T * √T * d) | Longformer, BigBird，长文档 |
| 线性注意力 | O(T * d^2) | Performer，理论上限 |
| 滑动窗口注意力 | O(T * W * d) | 局部依赖任务 |

> 注意：GPT 用的是「因果/自回归（causal）注意力」，
> 即位置 i 只能 attend 到 1..i 的位置（防止看未来）。

### 本章 vs 其他章节

| 章节 | 关键组件 | 适用数据类型 |
|------|---------|------------|
| 第二章（监督） | 线性 / 树 / KNN | 表格 |
| 第四章（MLP） | 全连接层 | 展平向量 |
| 第五章（CNN） | 卷积 + 池化 | 图像 |
| 第六章（RNN） | 循环状态 + 门 | 文本/语音/时间序列 |
| 第七章（Transformer） | 自注意力 + FFN | **通用**（文本/图像/语音/表格） |

> **经验**：Transformer 是「通用建模语言」，只要数据量够大、
> 算力够多，基本能在所有任务上达到 SOTA；但小数据、低算力场景下，
> LSTM、XGBoost、线性模型仍是「更划算」的选择。

---

## 三、运行方式

`
python main.py -m attention_demo -o output
python main.py -m transformer_demo -p 16,4,8 -o output   # d_model, heads, seq_len
`
