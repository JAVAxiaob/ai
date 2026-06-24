# 数字图像处理算法库

基于 Gonzalez《数字图像处理》第 1–12 章，实现教材中的主要图像处理方法，支持 JSON 流水线配置。

## 目录

| 路径 | 说明 |
|------|------|
| `code/` | Python 实现（按章节分子目录） |
| `md/` | 各章算法说明文档 |
| `img/` | 测试输入图像 |
| `imgout/` | 处理结果输出 |

## 快速开始

```bash
cd D:\job_files\图像算法\code
pip install -r requirements.txt
```

### JSON 流水线（总入口）

```bash
python main.py -i D:\job_files\图像算法\img\test1.png -c configs\demo_pipeline.json -o D:\job_files\图像算法\imgout\result.png
```

JSON 示例（按顺序执行）：

```json
{
  "直方图均衡化": "",
  "高斯滤波": "5, 1.0",
  "Canny边缘": "80, 160"
}
```

参数以逗号分隔；空字符串表示默认参数。方法名支持英文函数名或中文别名。

### 批量测试全部方法

```bash
python run_all_tests.py
```

对 `img/test1.png`、`img/test2.png` 各运行全部 **131** 个方法，结果保存至 `imgout/`。

### 列出所有方法

```bash
python main.py --list -i dummy -c dummy -o dummy
```

（需带占位参数；或 `python -c "from registry import METHOD_REGISTRY; print(len(METHOD_REGISTRY))"`）

## 章节与代码对应

| 章 | 目录 |
|----|------|
| 1 绪论 | `code/ch01_intro/` |
| 2 数字图像基础 | `code/ch02_basics/` |
| 3 灰度变换与空间滤波 | `code/ch03_spatial/` |
| 4 频率域滤波 | `code/ch04_frequency/` |
| 5 图像复原与重建 | `code/ch05_restoration/` |
| 6 彩色图像处理 | `code/ch06_color/` |
| 7 小波与多分辨率 | `code/ch07_wavelet/` |
| 8 图像压缩 | `code/ch08_compression/` |
| 9 形态学 | `code/ch09_morphology/` |
| 10 图像分割 | `code/ch10_segmentation/` |
| 11 表示与描述 | `code/ch11_representation/` |
| 12 目标识别 | `code/ch12_recognition/` |

详细说明见 `md/ch01.md` … `md/ch12.md`。
