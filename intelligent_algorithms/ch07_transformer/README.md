# 第七章：Transformer

> 面向对象：零基础小白
> 本章目标：理解自注意力机制、位置编码、Encoder-Decoder 架构。

---

## 一、从 RNN 到 Transformer：核心转变

2017 年 Google 发表《Attention is All You Need》，彻底改变了 NLP，后来扩散到视觉、语音、多模态。

**RNN 的两个根本问题**：
1. **串行计算无法并行**：h_t 依赖 h_{t-1} -> T 个时间步必须依次算 T 次
2. **长期依赖仍然困难**：即使有 LSTM，信息要经过 100+ 步「接力传播」才能从第一步传到最后一步

**Transformer 的核心洞察**：不用「按顺序读」，改为「看全部位置，然后加权组合」—— 自注意力（Self-Attention）。

---

## 二、Scaled Dot-Product Attention（缩放点积注意力）

### 1. 三个角色：Query, Key, Value

对每个位置 i：
- **Q_i（Query 查询）**：当前位置「想知道什么」
- **K_j（Key 键）**：每个其他位置「提供什么信息」
- **V_j（Value 值）**：每个其他位置「对应的具体内容」

**计算流程**：
`
step 1: 计算 Q 与每个 K 的相似度：score_j = Q_i * K_j （点积）
step 2: 除以 sqrt(d_k) 缩放（防止 d_k 大时点积过大 -> softmax 梯度消失）
step 3: softmax 归一化 -> alpha_1, alpha_2, ..., alpha_T （和为 1 的权重）
step 4: 按权重组合 V：output_i = sum_j alpha_j * V_j
`

**向量化形式**（一次算完所有位置）：
`
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V
`

**形状**：Q: [T_q, d_k]，K: [T_k, d_k]，V: [T_k, d_v] -> 输出: [T_q, d_v]

### 2. 一步步手算一个小例子

**设**：3 个词，每个词的 d_k = d_v = 2。

`
Q = [[1, 0],      (query for 词1)
     [0, 1],      (query for 词2)
     [1, 1]]      (query for 词3)

K = [[1, 0],      (key for 词1)
     [0, 1],      (key for 词2)
     [1, 1]]      (key for 词3)

V = [[1, 2],      (value for 词1)
     [3, 4],      (value for 词2)
     [5, 6]]      (value for 词3)

step 1: Q @ K^T = [[1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 2]]

step 2: 除以 sqrt(2) = 1.414 -> [[0.707, 0, 0.707],
                                   [0, 0.707, 0.707],
                                   [0.707, 0.707, 1.414]]

step 3: 每行 softmax
  第一行: [exp(0.707), exp(0), exp(0.707)] / sum
        = [2.028, 1, 2.028] / 5.056 = [0.40, 0.20, 0.40]
  第二行: 同理 = [0.20, 0.40, 0.40]
  第三行: 对词3 权重最高 = [0.22, 0.22, 0.56]

step 4: 加权 V
  output[0] = 0.40*[1,2] + 0.20*[3,4] + 0.40*[5,6] = [3.0, 4.0]
  output[1] = 0.20*[1,2] + 0.40*[3,4] + 0.40*[5,6] = [3.4, 4.4]
  output[2] = 0.22*[1,2] + 0.22*[3,4] + 0.56*[5,6] ≈ [3.68, 4.68]
`

**直觉解读**：每个输出位置都是**所有输入位置的加权和**，权重由 query-key 的相似度决定。在这个例子里，词1 看词1 和词3（权重各 0.40），因为 Q_1=[1,0] 匹配 K_1=[1,0] 和 K_3=[1,1] 的第一个维度。

### 3. 为什么要除以 sqrt(d_k)？

直觉：如果 d_k 很大（比如 d_k=512），Q_i 和 K_j 都是 512 维的均值 0 方差 1 的随机向量，它们的点积均值 0 方差 d_k -> 标准差 sqrt(d_k) -> 点积的大小约为 sqrt(512) ≈ 22。softmax 对 20 量级的输入会非常「尖锐」（一个接近 1，其他接近 0）-> 反向传播时梯度接近 0。

**除以 sqrt(d_k) 把点积的标准差拉回 1** -> softmax 输入在合理范围 -> 梯度正常流动。

这是一个简单但关键的工程修正（论文标题里的「Scaled」就是指这个）。

### 4. Mask（掩码）：让 Decoder 不偷看未来

在翻译任务中，生成第 t 个词时，模型只能看到前 t-1 个词，不能偷看第 t 个及以后（否则训练就像考试提前看了答案）。

**做法**：计算 Q@K^T 后，把右上三角（位置 j > i 的位置）设为 -inf -> softmax 后这些位置权重为 0。

### 5. Padding Mask：让 <PAD> 不参与注意力

和 RNN 一样，一个 batch 内的句子被 padding 到相同长度。对 <PAD> 位置的 key 设一个很大的负数（如 -1e9）-> softmax 后权重为 0 -> 模型不关注填充位置。
---

## 三、多头注意力（Multi-Head Attention）

### 1. 直觉：一个注意力头只看一种关系

单一的 Q/K/V 投影只能学到「一种」注意力模式。比如「it」这个词，可能同时需要关注：
- **指代关系**（和 animal 对齐）
- **因果关系**（和 tired 对齐）
- **语法关系**（和动词 was 对齐）

多头注意力让**不同的头关注不同的关系**。每个头有独立的 Q/K/V 投影矩阵，算出不同的注意力权重。

### 2. 计算流程

`
输入 X: [T, d_model]

对每个头 h (共 num_heads 个头):
    Q_h = X @ W_Q_h    [T, d_k]  其中 d_k = d_model / num_heads
    K_h = X @ W_K_h    [T, d_k]
    V_h = X @ W_V_h    [T, d_v]  通常 d_v = d_k
    head_h = Attention(Q_h, K_h, V_h)   [T, d_v]

拼接所有头:
concat(head_1, head_2, ..., head_H): [T, H * d_v] = [T, d_model]

最后做一次线性投影:
output = concat @ W_O   [T, d_model]
`

**论文默认设置**：d_model = 512，num_heads = 8，每个头 d_k = d_v = 64。

**为什么有效**：
- 每个头的 d_k 很小（64）-> 每个头只负责一部分维度的模式
- 不同头可以学到完全不同的注意力模式（如头1 学指代、头2 学语法、头3 学局部相邻）
- 最后用 W_O 把不同头学到的关系「融合」成一个输出

### 3. 注意力的三种用法（在 Transformer 中的三个地方）

| 位置 | Q 来自 | K, V 来自 | 作用 | mask 方式 |
|:---:|:---:|:---:|:---|:---|
| **Encoder self-attn** | 输入 | 输入 | 每个词看全部上下文 | padding mask |
| **Decoder self-attn** | 已生成的词 | 已生成的词 | 生成词时看之前生成的内容 | causal mask (不能看未来) + padding mask |
| **Decoder cross-attn** | Decoder 输出 | Encoder 输出 | 翻译时看原文选择要翻译的部分 | padding mask |

**Cross Attention（交叉注意力）**是翻译任务的关键：Decoder 在生成「le chat」时，用自己当前的 query 去看 Encoder 对「the cat」的 key，把相关信息拿过来。

---

## 四、位置编码（Positional Encoding）：让 Attention 知道顺序

**问题**：自注意力是「位置无关」的——它只看点积，不关心两个词在句子中的位置。如果把词顺序打乱，Attention 结果不变。但顺序对语言非常重要！

**解决方案**：给每个词的 embedding 加上一个「位置编码」pos_enc[i]，其中 i 是位置索引（0, 1, 2, ..., T-1）。

`
x_emb[i] = word_emb[word_i] + pos_enc[i]     (两个都是 d_model 维，直接相加)
`

### 原始论文的 sin/cos 位置编码：

`
pos_enc[pos, 2i]     = sin( pos / 10000^(2i / d_model) )
pos_enc[pos, 2i+1]   = cos( pos / 10000^(2i / d_model) )
`

**直觉**：
- 每个位置有一个独一无二的 d_model 维向量
- 不同频率的 sin/cos 让「相对位置」可以被线性表示（pos+k 的编码可以表示成 pos 编码的线性函数）
- 无需训练，固定公式 → 可以泛化到比训练集更长的句子

### 可学习的位置编码（Learned Positional Embedding）

更常见的现代做法（BERT、GPT 系列）：直接把位置当作另一个「词」，给每个位置学一个 embedding。

`
pos_emb = Embedding(num_positions, d_model)    (需要学习的参数)
x_emb[i] = word_emb[word_i] + pos_emb[i]
`

**对比**：

| 方式 | 是否可训练 | 能否泛化到更长序列 | 效果 |
|:---:|:---:|:---:|:---|
| sin/cos | 否 | 能（公式通用）| 在 Transformer 原文中稍好 |
| learned | 是 | 不能（超过 num_positions 就没定义了）| 在 BERT/GPT 等后续工作中更常用，实际效果好 |

**现代常见做法**：用 learned positional embedding，配合 RoPE（旋转位置编码，GPT-Neo、LLaMA 采用）或 ALiBi（无位置编码，靠注意力衰减模拟位置）等更先进的位置方案。

### 为什么用「加法」而不是「拼接」？

直觉：如果用拼接 [word_emb; pos_enc]，模型需要学「如何把前半和后半组合」，参数更多。用加法，相当于让模型在同一个表示空间里同时编码「语义」和「位置」，模型可以学到「某些维度更关心语义，某些维度更关心位置」。这是一个工程选择（减少参数 + 效果不差）。

---

## 五、一个 Transformer Block 的完整结构

### Encoder block（论文有 N=6 个相同的 block 堆叠）

`
输入 X (形状: [T, d_model])
   |
   v
Multi-Head Self-Attention   (每个位置看所有位置)
   |
   v
X + Attention(X)     -> 残差连接（ResNet 的 skip connection）
   |
   v
LayerNorm(...)       -> 层归一化（把每个位置的向量归一化到均值0方差1）
   |
   v
Position-wise FFN    -> 对每个位置独立做: Linear(d_model -> 4*d_model) -> ReLU -> Linear(4*d_model -> d_model)
   |
   v
X + FFN(X)           -> 残差连接
   |
   v
LayerNorm(...)       -> 层归一化
   |
   v
输出 Y (形状: [T, d_model])
`

**关键组件解释**：

1. **残差连接 x + f(x)**：和 ResNet 一样。让深层网络可训练。梯度可以直接走「x」那条通路，不经过 f 的参数层。

2. **Layer Normalization**：对每个 token 的 d_model 维做归一化（减去均值、除以标准差），再加一个可学习的缩放 gamma 和偏移 beta。和 BatchNorm 不同：LayerNorm 沿特征维度归一化，不依赖 batch 大小，适合变长序列。

   **位置（Pre-norm vs Post-norm）**：
   - Post-norm（原始论文）: LayerNorm(x + Sublayer(x))
   - Pre-norm（现代主流）: x + Sublayer(LayerNorm(x)) -> 更稳定，特别是深层大模型。现在几乎所有大模型都用 Pre-norm。

3. **Position-wise Feed-Forward Network (FFN / MLP)**：对每个位置独立做两层 MLP。这一层让「不同维度的信息可以交互」（注意力让「不同位置的信息可以交互」）。两者互补。

   中间维度通常是 4*d_model（论文默认）= 2048。MLP 占模型算力和参数的大头（远大于注意力）。

### Decoder block（同样 N=6 个堆叠）

在 Encoder block 基础上多加一个 **Cross Attention**：

`
输入 Y（已生成的部分）
   |
   v
Masked Multi-Head Self-Attention  (加 causal mask，不能看未来位置)
   |
   v
残差 + LayerNorm
   |
   v
Multi-Head Cross Attention        (Q 来自 Decoder，K/V 来自 Encoder 输出)
   |
   v
残差 + LayerNorm
   |
   v
Position-wise FFN
   |
   v
残差 + LayerNorm
   |
   v
输出
`

**训练时**：Decoder 一次性接收完整的目标句子（如法语整句），用 causal mask 保证每个位置只能看自己之前的内容，一次并行算出所有位置的预测 → 和真实词做交叉熵。这比 RNN 快一个数量级。

**推理时（自回归生成）**：每次只生成一个词，把刚生成的词追加到输入，再跑一次 forward。所以推理是串行的（O(T) 次 forward），但每次 forward 很快（并行处理已有内容）。

### 整体架构一览（机器翻译场景）

`
英文句子 -> Embedding + Positional Encoding
   |
   v
N x Encoder Blocks -> 编码后的表示 memory: [T_enc, d_model]
   |
   v
法语（已经生成的部分）-> Embedding + Positional Encoding
   |
   v
N x Decoder Blocks (每个 block 有 self-attn + cross-attn + FFN) -> output: [T_dec, d_model]
   |
   v
Linear(d_model -> vocab_size) + softmax -> 下一个词的概率分布: [T_dec, |V|]
`

**分类/特征提取场景（如 BERT）**：只有 Encoder，无 Decoder。在句子开头加一个特殊标记 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>，用 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 的输出接 Linear 做分类。

**语言模型场景（如 GPT）**：只有 Decoder（因果自注意力），无 Encoder。每次生成一个词，自回归地生成文本。
---

## 六、运行方式

```bash
python main.py -m self_attention_demo -p 5,12,8 -o output   # T=5, d_model=12, num_heads=8
python main.py -m transformer_demo -p 5,12,8,2 -o output    # 外加 ff_dim=2048 的简单 Transformer
```

运行后可以看到：
- 自注意力权重矩阵的热力图（每个位置对其他位置的权重）
- Transformer 前向传播的各层形状变化

---

## 七、问题与答案解析

### Q1：自注意力为什么比 RNN 更擅长处理长期依赖？

**答**：RNN 中信息从位置 1 传到位置 T 必须经过 T 步「接力」（每步要乘一次 W_hh）。任何一步有信息衰减，最终就传不到。

自注意力中，**任意两个位置之间只有一步连接**—— Query 直接和 Key 做点积，不管它们相隔多少个词。梯度可以无衰减地从 T 直接传回 1。

**量化对比**：从位置 i 到位置 j 的「最大路径长度」：
- RNN: |i-j| 步（线性距离）
- Self-Attention: 1 步（常数）
- CNN（k=3）: ceil(|i-j|/2) 步（对数级？不，仍然线性，但系数更小）

---

### Q2：自注意力的计算复杂度是多少？和 RNN/CNN 对比？

**答**：设序列长度为 T，维度为 d。

| 方法 | 每步复杂度 | 并行度 | 最大路径长度 |
|:---:|:---:|:---:|:---:|
| RNN | O(T * d^2) | 1 (串行) | O(T) |
| CNN (k=3) | O(T * d^2 * k) | T | O(log_k T) |
| Self-Attention | O(T^2 * d) | T | O(1) |

**关键观察**：当 T < d 时（短序列），自注意力的复杂度低于 RNN/CNN。当 T > d 时（长序列，如 T=10000），T^2 主导，自注意力会变得非常慢 -> 需要稀疏注意力 / 局部注意力 / FlashAttention 等优化。

---

### Q3：为什么 Transformer 仍然需要位置编码？如果去掉会怎样？

**答**：自注意力本身是「置换不变（permutation invariant）」的——即 f(x_1,...,x_T) = f(x_{shuffled(1..T)}) 如果你没有位置信息。句子「我 爱 你」和「你 爱 我」在纯自注意力下会得到完全一样的输出！

位置编码让模型知道「这个词在哪个位置」，从而可以学到「位置相关的模式」（比如「主语通常在动词前面」）。

**去掉位置编码的效果**：在需要顺序信息的任务上（翻译、命名实体识别）性能大幅下降；在纯词袋任务（如情感分类）下降较少。

---

### Q4：为什么多头注意力比单头好？一个头不够用吗？

**答**：一个头只能学「一种」注意力模式（比如一种语法关系）。多个头让模型同时关注多种关系：

- 头1 关注「指代」（it -> animal）
- 头2 关注「动宾搭配」（eat -> food）
- 头3 关注「局部相邻词」（类似 n-gram）
- 头4 关注「远距离依赖」（...）

**数学角度**：多头让注意力可以在**多个子空间**中独立进行。每个子空间学到不同的关系类型，最后用 W_O 拼起来。实验显示（论文和后续工作）8 头或 16 头是较好的折中。

**极端**：d_k=1 且 num_heads=d_model 的情况是「多维度注意力」（每个维度独立做一次注意力），有时效果也不错。但通常的做法是 d_k=d_model/num_heads=64。

---

### Q5：Encoder 和 Decoder 有什么区别？什么时候用哪个？

**答**：

| 组件 | 注意力方式 | 典型用例 | 代表模型 |
|:---:|:---|:---|:---|
| **只使用 Encoder** | 双向 self-attention（每个位置看全部）| 分类、命名实体识别、句子嵌入 | BERT、RoBERTa |
| **只使用 Decoder** | 因果 self-attention（每个位置只能看之前）| 语言模型、文本生成 | GPT-1/2/3、LLaMA、GPT-4 |
| **Encoder-Decoder** | Encoder 双向 + Decoder 因果 + cross-attn | 机器翻译、摘要、语音识别 | T5、BART、Whisper |

**选择准则**：
- 如果你的任务是「理解一段文本，输出一个标签/向量」-> 用 Encoder-only（BERT 风格）
- 如果你的任务是「继续写文本」-> 用 Decoder-only（GPT 风格）
- 如果你的任务是「把一段文本变成另一段文本（输入输出都长，且结构不同）」-> 用 Encoder-Decoder（T5 风格）

**趋势**：2022 年以后，Decoder-only 大模型（LLaMA、GPT-4 等）成为主流，因为「一个大的因果语言模型可以通过 prompting 做几乎所有任务」。但在特定领域（如语音识别 Whisper、翻译）Encoder-Decoder 仍有优势。

---

### Q6：为什么 Transformer 需要 LayerNorm 和残差连接？去掉会怎样？

**答**：两者配合让深层 Transformer 可以稳定训练。

**残差连接 x + f(x)**：让深层网络至少不比浅层网络差（最坏情况 f(x)=0，恒等映射）。更重要的是——反向传播时梯度有一条「高速公路」（直接从 loss 传到 x，不经过 f 的权重），避免了深层网络的梯度消失问题。

**Layer Normalization**：把每个 token 的特征向量归一化到均值 0、方差 1，然后通过可学习的 gamma/beta 再缩放。作用是**稳定激活值范围**，让后续层的输入分布在合理区间。

**去掉残差**：深度 > 10 层后训练不稳定甚至发散（梯度消失）。
**去掉 LayerNorm**：激活值逐层放大/缩小 -> 数值不稳定。

**Pre-norm vs Post-norm 提醒**：原始论文用 Post-norm（LN(x + f(x))），现代大模型几乎都用 Pre-norm（x + f(LN(x))）。Pre-norm 更稳定，因为主通路（x 通路）上没有非线性变换，梯度可以顺利流通。

---

### Q7：Position-wise FFN 是做什么的？为什么在 Attention 之后还需要？

**答**：Attention 让「不同位置的信息可以交互」（空间/序列维度的混合）。FFN 让「同一位置的不同特征维度可以交互」（特征维度的混合）。两者互补！

**具体来说**：
- Attention 学到了「词 A 应该吸收词 B 和词 C 的信息」，但它对每个维度使用相同的权重 alpha
- FFN 学到了「在吸收了这些信息后，这个词的第 37 维应该被放大，第 128 维应该被清零」——在特征维度做非线性变换

**结构**：Linear(d_model -> 4*d_model) -> ReLU/GELU -> Linear(4*d_model -> d_model)
中间维度是 4*d_model（论文默认）。FFN 占模型参数和算力的大头（~2/3）。

**如果去掉 FFN**：可以把它理解为「只做信息重新分布、不做特征变换」的模型——表达能力大幅下降，实际任务上效果差很多。

---

### Q8：为什么位置编码用加法不用拼接？拼接不丢失信息吗？

**答**：直觉上拼接 ([word_emb; pos_enc]) 似乎「保留了更多信息」，但实际上：

1. **模型可以学到分离信息**：如果某些维度在 word_emb 中永远是语义信息，某些维度在 pos_enc 中永远是位置信息，模型可以通过后续的 Linear 层自动学到哪些维度关心哪部分信息。用加法只是「在同一个空间里混合」，模型自己会解耦。

2. **参数更少**：拼接后维度变成 2*d_model，后续的 Attention weight Q/K/V 矩阵从 [d_model, d_model] 变成 [2*d_model, 2*d_model]，参数翻 4 倍。加法保持同样的维度。

3. **实验证明效果不差**：论文作者实验了加法和拼接，加法效果略好（可能是因为参数更少，正则化效果更强）。

**一个类比**：想象你在做照片分类（RGB 三个通道）。三个通道是「拼接」成 [H, W, 3] 的，因为它们本质上是不同的模态。但对于「词身份」和「位置」，模型可以在同一个表示空间里同时编码这两种信息——所以加法够用了。

---

### Q9：自注意力的 softmax 权重总是稀疏的吗？它们具体长什么样？

**答**：不一定。取决于层深度、任务和具体的头。

**BERT-base 的实验观察**（参考原论文和后续分析）：

- **底层（layer 1-2）**：大多数头关注局部邻域（附近 1-3 个词）。有些头学到了类似 n-gram / 短语边界的模式。权重相对集中。
- **中间层（layer 3-8）**：开始出现更长距离的依赖。一些头专门关注「动词 -> 主语」「介词 -> 名词」等句法关系。
- **顶层（layer 9-12）**：更全局、更抽象的注意力模式。有的头几乎均匀分布（「看所有词」），有的头专注于特殊 token（<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>、[SEP]）。

**通常不是极度稀疏**——除了专注于某个词的头之外，大多数头的权重分布在 5-20 个词上，不是一个独热向量。

---

### Q10：Transformer 和 CNN 相比，在视觉上谁更强？

**答**：这个问题至今没有绝对答案，研究在持续发展。

**2020 年以前**：CNN（ResNet、EfficientNet 等）在图像分类、检测、分割上是绝对主流。Transformer 在 NLP 是绝对主流，但在视觉上还没证明自己。

**2020-2022**：Vision Transformer (ViT, Dosovitskiy et al.) 出现——把图像切成 16x16 的 patch，当成「词」，送入标准 Transformer。在大数据（JFT-300M 图片）预训练后，ViT 超过所有 CNN！但在中等数据量（ImageNet 1.2M）上 CNN 仍然稍好（CNN 的归纳偏置很适合视觉，不需要那么多数据学习）。

**现在（2024+）**：CNN 借鉴 Transformer（ConvNeXt：把 ResNet 改成 Transformer 的训练方式/设计 -> 反超 ViT），Transformer 借鉴 CNN（Swin Transformer：层级化 + 局部注意力，更像 CNN 的设计）。两者在设计上趋同。**大模型大数据下 Transformer 强**；**小模型小数据下 CNN 强**。

---

### Q11：训练 Transformer 有什么关键技巧？

**答**：Transformer 对初始化、学习率、正则化都很敏感。以下是公认的「默认最佳实践」：

1. **Warmup 学习率**：前 N 步线性把 lr 从 0 升到最大值，再用余弦退火衰减。Transformer 对初始 lr 敏感，小 lr 的 warmup 让 Q/K/V 等权重先进入合理范围。

2. **AdamW 优化器**（Loshchilov & Hutter, 2019）：Adam + 解耦的权重衰减（weight decay）。比 vanilla Adam 对正则化更友好。默认 weight_decay=0.01。

3. **合适的 beta1/beta2**：beta1=0.9, beta2=0.999（论文）或 beta2=0.98（大模型）。

4. **初始化**：Linear 权重用 Xavier / N(0, 1/sqrt(d_model)) 初始化。某些特定位置（如 Decoder 最后的 Linear 输出 logits）可能需要更小的初始化。

5. **Label Smoothing**：把 one-hot 目标变成 0.9 + 0.1/num_classes 的「软」目标。防止模型过度自信。默认 smoothing=0.1。

6. **Dropout**：在 Embedding、注意力、FFN 后加。小模型 dropout=0.1，大模型更小（0.0 或 0.05）。

7. **Gradient Clipping**：max_norm=1.0，防止训练初期梯度爆炸。

8. **Layer order**：Pre-norm（LN 在子层前面，残差后不加 LN）比 Post-norm（论文原始）更稳定。现在大模型都用 Pre-norm。

9. **激活函数**：GELU（Hendrycks & Gimpel, 2016）代替 ReLU。在激活为 0 时梯度为 0.5（不是 0），让梯度更平滑。

---

### Q12：FlashAttention 是什么？为什么它能让大模型训练快 2-4 倍？

**答**：FlashAttention 是 2022 年由 Tri Dao 等人提出的一个工程优化，目标是**在 GPU 上更快、更省显存地计算 Attention**。

**问题**：标准 Attention 的显存瓶颈是 S = Q @ K^T / sqrt(d_k)，形状 [T, T]。当 T=8192 时 S 是 8192x8192 ≈ 67M 元素（每步 float16 需要 134MB，但还要存 softmax 的中间量）。在多层堆叠下，每一层都要保存 S 来做反向传播 -> 显存爆炸。

**FlashAttention 的核心思想**：
- 把 Q/K/V 分成小块（blocks），在 GPU SRAM（高速但小）内做小矩阵乘法，用「online softmax」技巧（不需要保存完整 S 就能算出 softmax 的正确输出）
- 只对需要的 block 重新计算 -> 不保存完整的 S 矩阵 -> 大幅减少显存和 HBM 访问
- 最终 Attention 更快 + 显存线性 O(T) 而不是 O(T^2)

**结果**：训练一个 175B 参数 GPT-3 级别的模型，FlashAttention 让训练时间从几个月降到几周。**没有改变任何数学，只有实现层面的优化——所以是零代价加速**。现在 PyTorch 2.0 的 F.scaled_dot_product_attention 已经内置了类似优化。

---

## 本章小结

1. **自注意力**：Q/K/V 三组投影 -> 每个位置对所有位置加权组合。任意两个位置之间只有一步连接 -> 解决长依赖。
2. **为什么除以 sqrt(d_k)**：防止大 d_k 下点积过大 -> softmax 饱和 -> 梯度消失。
3. **Mask 两种**：Causal mask（让 Decoder 不看未来）和 Padding mask（让 <PAD> 不参与注意力）。
4. **多头注意力**：在多个子空间独立做注意力，学到不同关系类型（指代、句法、长依赖等）。
5. **位置编码**：sin/cos 固定编码或可学习 embedding。现代大模型常用 RoPE、ALiBi 等更先进的位置方案。
6. **完整 Transformer Block**：Attention -> 残差 -> LayerNorm -> FFN(4*d_model 中间层) -> 残差 -> LayerNorm。
7. **Encoder vs Decoder**：Encoder 双向 + 全局视野（适合理解任务），Decoder 因果 + 自回归生成（适合生成任务），Encoder-Decoder 两者结合（适合翻译、摘要等「输入-输出都长」的任务）。
8. **计算复杂度**：注意力 = O(T^2 * d)。短序列友好，长序列需要 FlashAttention 等优化。
9. **训练技巧**：AdamW + warmup + 余弦退火、Label Smoothing、Pre-norm、GELU 激活、Gradient Clipping。
10. **实际部署**：用 PyTorch 2.0 的 F.scaled_dot_product_attention 或 FlashAttention 获得 2-4 倍加速。