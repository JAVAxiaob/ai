# 第六章：循环神经网络 RNN / LSTM

> 面向对象：零基础小白
> 本章目标：理解「序列数据」的建模方法，从 Vanilla RNN 走到 LSTM。

---

## 一、章节知识点

### 6.1 RNN 原理

序列数据（时间序列、文本）有一个重要特点：**t 时刻的状态依赖 t-1 时刻**。
RNN 用一个「隐藏状态 h」携带历史信息：

`
h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b)
`

**问题**：长序列下，梯度多次乘 W_hh → 要么爆炸（>1 连乘）要么消失（<1 连乘），
远距离信息学不到。

对应代码：
nn_demo(seq_len=10, input_dim=4, hidden=8)
- NumPy 手动实现单步 RNN 前向；
- 隐藏状态热力图见 output/ch06_rnn.png。

### 6.2 LSTM（Long Short-Term Memory）

LSTM 引入**细胞状态 c** 和 3 个门控（gate）来选择性保留信息：

| 门 | 公式 | 作用 |
|----|------|------|
| 遗忘门 f | σ(W_f @ [h_{t-1}, x_t] + b_f) | 控制旧状态保留多少 |
| 输入门 i | σ(W_i @ ...) | 控制新候选信息写入多少 |
| 候选 c̃ | tanh(W_c @ ...) | 新状态候选 |
| 输出门 o | σ(W_o @ ...) | 控制输出多少 c_t |

**更新**：
`
c_t = f ⊙ c_{t-1} + i ⊙ c̃_t
h_t = o ⊙ tanh(c_t)
`
「遗忘门 ≈1 + 输入门 ≈0」的组合让 c 可以长时间稳定携带信息，
**解决了 RNN 的梯度消失问题**。

对应代码：lstm_demo(seq_len=10, input_dim=4, hidden=8)。
可视化见 output/ch06_lstm.png。

> GRU 是 LSTM 的简化版：把遗忘门和输入门合并为「更新门 z」，
> 把细胞状态 c 和隐藏状态 h 合并。参数量更少，速度更快，效果接近。

### 6.3 文本处理示例

文本 = 序列数据。一个最小的「BoW + 朴素贝叶斯」文本分类流水线：

1. **分词/词表**：CountVectorizer（sklearn）将每段文本转换成「词频向量」；
2. **分类器**：MultinomialNB（多项式朴素贝叶斯）在词频向量上预测类别；
3. **评估**：准确率。

对应代码：	ext_classify_demo()，构造了 2 类（ML vs 体育）短文本语料。

---

## 二、横向对比

### RNN vs LSTM vs GRU

| 维度 | Vanilla RNN | LSTM | GRU |
|------|------------|------|-----|
| 核心组件 | 只有 h | h + c + 3 个门 | 只有 h + 2 个门 |
| 梯度消失 | ❌ 严重 | ✅ 大幅缓解 | ✅ 较好 |
| 参数量 | O((d+h)*h) | O(4*(d+h)*h) | O(3*(d+h)*h) |
| 训练速度 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| 长序列表现 | ❌ | ✅ | ✅ |
| 何时用 | 教学/短序列 | 基准，强基线 | 参数量受限时优先 |

### 文本处理方法演进

| 方法 | 代表 | 原理 | 优点 | 缺点 |
|------|------|------|------|------|
| 词袋 (BoW) | CountVectorizer + NB | 统计词频 | 快、简单、可解释 | 忽略顺序，「我恨你」=「你恨我」 |
| TF-IDF | TfidfVectorizer | 词频 × 逆文档频 | 稍好于 BoW | 仍忽略顺序 |
| Word2Vec | skip-gram / CBOW | 学习稠密词嵌入 | 相似词向量接近 | 静态嵌入，一词多义问题 |
| RNN/LSTM | seq2seq | 序列建模 | 捕捉顺序 | 慢，难以并行 |
| Transformer | BERT, GPT | 自注意力 | **并行 + 长距离依赖** | 需要大算力 |

> 注意：本章文本示例仍用 BoW+NB（零基础友好 + 不依赖外部库），
> 真正的工业级文本模型现在几乎全部是 Transformer 家族。

### 本章 vs 第五章 (CNN)

| 维度 | CNN | RNN/LSTM |
|------|-----|---------|
| 输入形式 | 2D 图像 (H,W,C) | 1D 序列 (T,d) |
| 归纳偏置 | **空间局部性** | **时间顺序性** |
| 并行性 | ✅ 非常容易并行（每层无依赖） | ❌ 串行，必须 t→t+1 |
| 长距离依赖 | 靠多层堆叠实现 | 理论上有 LSTM 门控 |
| 典型任务 | 图像分类、目标检测 | 文本、语音、时间序列 |

---

## 三、运行方式

`
python main.py -m rnn_demo -p 10,4,8 -o output     # seq_len=10, input_dim=4, hidden=8
python main.py -m lstm_demo -p 10,4,8 -o output
python main.py -m text_classify_demo -o output
`
