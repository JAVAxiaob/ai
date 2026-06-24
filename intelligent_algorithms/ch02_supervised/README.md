# 第二章：监督学习

> 面向对象：零基础小白
> 本章目标：掌握5种最常用的监督学习算法，理解它们的适用场景与数学原理。

---

## 一、章节知识点

### 2.1 线性回归（Linear Regression）

**核心思想**：用一条直线拟合 y 和 X 的关系：y = X @ θ。

**解析解（OLS，最小二乘法）**：
`
θ = (X^T X)^{-1} X^T y
`
本项目 linear_regression_demo() 即使用 NumPy 手动实现这个公式。

### 2.2 逻辑回归（Logistic Regression）

**核心思想**：在「线性组合 + sigmoid」后做二分类。
`
p = sigmoid(X @ w + b)  # 输出 [0,1] 概率
L = -mean(y*log(p) + (1-y)*log(1-p))  # 交叉熵损失
`
**调用**：logistic_regression_demo() 使用 sklearn.linear_model.LogisticRegression。

### 2.3 决策树（Decision Tree）

**核心思想**：从根节点开始，每次选「最能区分类别」的特征做一次分裂，
形成一棵 if-else 树。优点是可解释性强、支持非线性。

**调用**：decision_tree_demo() 使用 sklearn.tree.DecisionTreeClassifier(max_depth=4)。

### 2.4 K近邻（KNN）

**核心思想**：懒学习，预测时找 K 个最近的邻居投票。
`
predict(x) = vote( y_i for (x_i, y_i) ∈ K-nearest(x) )
`
**特点**：训练快、预测慢；对高维数据「维度灾难」敏感。
**调用**：knn_demo(k=5) 使用 sklearn.neighbors.KNeighborsClassifier。

### 2.5 朴素贝叶斯（Naive Bayes）

**核心思想**：基于贝叶斯定理 +「特征独立」假设。
`
P(y|X) ∝ P(y) * Π P(x_i|y)
`
**优点**：训练极快，在文本、医学数据上表现好；
**缺点**：「特征独立」假设在真实数据上通常不成立。
**调用**：
aive_bayes_demo() 使用 sklearn.naive_bayes.GaussianNB。

---

## 二、横向对比：5种监督学习方法

| 方法 | 模型形式 | 训练速度 | 预测速度 | 可解释性 | 对特征缩放敏感 | 对高维友好 |
|------|---------|---------|---------|---------|---------------|-----------|
| 线性回归 | 线性 θ | ⭐⭐⭐⭐ 解析解 O(n*d^2) | ⭐⭐⭐⭐⭐ O(d) | ⭐⭐⭐⭐⭐ 权值直接可读 | **是**，需标准化 | 一般（过拟合） |
| 逻辑回归 | sigmoid(线性) | ⭐⭐⭐⭐ 迭代优化 | ⭐⭐⭐⭐⭐ O(d) | ⭐⭐⭐⭐ 权值可解释 | **是** | 一般（需正则） |
| 决策树 | 分层 if-else | ⭐⭐⭐ O(n*log n) | ⭐⭐⭐⭐ O(depth) | ⭐⭐⭐⭐ 可画图 | 否 | ⭐⭐⭐ 易过拟合 |
| KNN | 惰性/距离投票 | **0**（无需训练） | ⭐ O(n*d) 慢 | ⭐⭐ 距离直觉 | **是**，需归一化 | ❌ 维度灾难 |
| 朴素贝叶斯 | 概率/贝叶斯 | ⭐⭐⭐⭐ O(n*d) | ⭐⭐⭐⭐ | ⭐⭐⭐ 条件概率可查 | 不太敏感 | ⭐⭐⭐ 文本仍可用 |

### 方法之间的互补关系

1. **线性 / 逻辑回归 vs 决策树**：
   - 线性模型对「线性关系」数据非常高效、可解释；
   - 决策树对「非线性、有交互作用」的数据更擅长，但容易过拟合。
   - 实际工程：**梯度提升树（GBDT / XGBoost）** 是决策树的强力升级版。

2. **KNN vs 其他所有**：
   - KNN 不训练，适合小数据、冷启动；
   - 但预测时要和所有样本比较距离，**数据 >10 万时基本不可用**。

3. **朴素贝叶斯 vs 逻辑回归**：
   - 朴素贝叶斯：小样本、特征独立时表现强，**生成式模型**（学习 P(X|y)）；
   - 逻辑回归：直接学 P(y|X)，**判别式模型**，工程上更常用。

### 本章 vs 第四章（深度学习）

- **本章**：浅层模型 + 强解释性 + 对小数据友好；
- **第四章**：深层神经网络 + 自动特征提取 + 需要大量数据/算力；
- **经验法则**：结构化表格数据（风控、营销）先用 GBDT，
  图像/文本/语音等非结构化数据再上深度学习。

---

## 三、运行方式

`
python main.py -m linear_regression_demo -o output
python main.py -m logistic_regression_demo -o output
python main.py -m decision_tree_demo -o output
python main.py -m knn_demo -p 7 -o output        # k=7
python main.py -m naive_bayes_demo -o output
`
