# 第五章：卷积神经网络 CNN

> 面向对象：零基础小白
> 本章目标：理解「卷积、池化」的直觉，实现手动卷积与简易 CNN。

---

## 一、章节知识点

### 5.1 CNN 原理

CNN 核心贡献：**局部感受野 + 参数共享 + 池化**。
- 局部感受野：每个神经元只看图像的一小块（3x3、5x5）；
- 参数共享：同一块卷积核在整张图上滑动，参数量骤降；
- 池化：把临近像素合并（取最大/平均），引入平移不变性。

因此 CNN 对图像这种「空间结构强」的数据，远优于 MLP。

### 5.2 卷积与池化（手动实现）

**卷积（Convolution）**：
`
out[i,j] = Σ (patch * kernel)  # 逐元素乘再求和
`
本章演示 4 种核：
| 核 | 效果 |
|----|------|
| Sobel x | 检测垂直边缘 |
| Sobel y | 检测水平边缘 |
| Mean 3x3 | 均值模糊 |
| Sharpen | 锐化（增强高频） |

**池化（Pooling）**：
- Max pooling：取窗口内最大值 → 保留最强响应；
- Average pooling：取窗口内平均值 → 更平滑。

对应代码：conv2d_demo()、pooling_demo()。
可视化分别见 output/ch05_conv2d.png 与 ch05_pooling.png。

### 5.3 经典 CNN 结构（概念介绍）

本项目的 cnn_demo() 使用「手动卷积提取边缘特征 + MLP 分类」来模拟一个简易 CNN
（真实生产环境会用 PyTorch/TensorFlow 搭真正的 CNN）。

经典模型对比：

| 模型 | 年份 | 核心贡献 |
|------|------|---------|
| LeNet-5 | 1998 | 最早的实用 CNN，LeCun |
| AlexNet | 2012 | ReLU + Dropout + GPU，ImageNet 突破 |
| VGG | 2014 | 统一 3x3 小卷积核，深网络 |
| GoogLeNet | 2014 | Inception 模块，多尺寸卷积并行 |
| ResNet | 2015 | **残差连接**，让 100+ 层成为可能 |

---

## 二、横向对比

### MLP vs CNN

| 维度 | MLP（展平为向量） | CNN（保留 2D 结构） |
|------|-----------------|-------------------|
| 输入形状 | (H*W*C,) 一维 | (H, W, C) 三维 |
| 参数数量 | O(H*W*hidden) 极大 | O(C_in*C_out*k^2) 小 |
| 空间先验 | ❌ 完全无 | ✅ 局部、平移不变 |
| 对图像任务 | ❌ 差 | ✅ 强 |
| 泛化到位置 | 差（像素移动导致完全不同输入） | 强（卷积核滑动） |

### 卷积核大小对比

| 核大小 | 感受野增长 | 参数 | 直觉 |
|--------|----------|------|------|
| 3x3 | 慢（需要多层堆叠） | 9 | 小核 + 多层 = VGG 经验 |
| 5x5 | 中 | 25 | 早期 AlexNet 使用 |
| 7x7 | 快 | 49 | ResNet 第一层下采样 |
| 1x1 | 不变（只跨通道） | 1 | 降维/升维（inception） |

### 不同池化方式对比

| 方式 | 公式 | 适用场景 |
|------|------|---------|
| Max Pool | max(window) | 保留纹理边缘，CNN 首选 |
| Avg Pool | mean(window) | 更平滑，小数据好 |
| Global Avg Pool | 整个 feature map 求平均 | 替代全连接层，减少参数 |

### 本章 vs 其他章节

- **vs 第二章（浅层）**：CNN 能自动学习特征 → 工程上更准；
- **vs 第六章（RNN）**：CNN 擅长空间结构，RNN 擅长时间顺序；
- **vs 第七章（Transformer）**：ViT（Vision Transformer）在大图像分类上已超越 CNN，但 CNN 仍是视觉任务的基线。

---

## 三、运行方式

`
python main.py -m conv2d_demo -o output
python main.py -m pooling_demo -o output
python main.py -m cnn_demo -p 50 -o output    # 50 epoch
`
