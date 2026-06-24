# -*- coding: utf-8 -*-
"""根据注册表生成各章节 Markdown 文档。"""

import inspect
import os
import sys

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODE_DIR)

from registry import ALIASES, METHOD_REGISTRY

MD_DIR = r"D:\job_files\图像算法\md"

CHAPTER_MAP = {
    "application_": 1,
    "simulate_": 1,
    "annotate_": 1,
    "resize_": 1,
    "downsample": 2,
    "quantize_": 2,
    "mark_": 2,
    "connectivity_": 2,
    "optical_": 2,
    "spectrum_": 2,
    "distance_transform_vis": 2,
    "convolve_": 2,
    "probability_": 2,
    "add_gaussian_noise": 2,
    "linear_": 3,
    "contrast_": 3,
    "log_": 3,
    "power_": 3,
    "threshold_": 3,
    "histogram_": 3,
    "local_histogram": 3,
    "mean_filter": 3,
    "gaussian_filter": 3,
    "median_filter": 3,
    "sobel_gradient": 3,
    "laplacian_sharpen": 3,
    "highpass_": 3,
    "unsharp_": 3,
    "fuzzy_": 3,
    "dft_": 4,
    "ideal_": 4,
    "butterworth_": 4,
    "gaussian_low": 4,
    "gaussian_high": 4,
    "band": 4,
    "homomorphic": 4,
    "fft_filter": 4,
    "add_periodic": 5,
    "add_salt": 5,
    "motion_": 5,
    "defocus_": 5,
    "mean_denoise": 5,
    "median_denoise": 5,
    "adaptive_denoise": 5,
    "wiener_": 5,
    "inverse_": 5,
    "constrained_": 5,
    "blind_": 5,
    "radon_": 5,
    "rgb_to_hsi": 6,
    "rgb_to_cmy": 6,
    "rgb_": 6,
    "color_": 6,
    "pseudocolor": 6,
    "dwt_": 7,
    "pyramid_": 7,
    "wavelet_": 7,
    "arithmetic_": 8,
    "video_frame": 8,
    "huffman_": 8,
    "run_length": 8,
    "lzw_": 8,
    "dct_": 8,
    "jpeg": 8,
    "erosion": 9,
    "dilation": 9,
    "opening": 9,
    "closing": 9,
    "hit_or_miss": 9,
    "boundary_": 9,
    "skeleton_": 9,
    "thinning": 9,
    "thickening": 9,
    "distance_transform": 9,
    "grayscale_": 9,
    "top_hat": 9,
    "black_hat": 9,
    "morphological": 9,
    "sobel_edge": 10,
    "prewitt_": 10,
    "laplacian_edge": 10,
    "canny_": 10,
    "otsu_": 10,
    "adaptive_threshold": 10,
    "multi_threshold": 10,
    "region_": 10,
    "watershed_": 10,
    "combined_": 10,
    "marked_graph": 11,
    "chain_": 11,
    "polygon_": 11,
    "texture_": 11,
    "pca_": 11,
    "harris_": 11,
    "sift_": 11,
    "surf_": 11,
    "orb_": 11,
    "template_": 12,
    "minimum_": 12,
    "bayesian_": 12,
    "svm_": 12,
    "shape_": 12,
    "cnn_": 12,
    "bp_neural": 12,
    "syntactic_": 12,
    "graph_matching": 12,
}

EXACT_CHAPTER = {
    "threshold_global": 10,
}


def guess_chapter(name: str) -> int:
    if name in EXACT_CHAPTER:
        return EXACT_CHAPTER[name]
    for prefix, ch in CHAPTER_MAP.items():
        if name.startswith(prefix) or name == prefix.rstrip("_"):
            return ch
    return 0


CHAPTER_TITLES = {
    1: "第1章 绪论",
    2: "第2章 数字图像基础",
    3: "第3章 灰度变换与空间滤波",
    4: "第4章 频率域滤波",
    5: "第5章 图像复原与重建",
    6: "第6章 彩色图像处理",
    7: "第7章 小波和多分辨率处理",
    8: "第8章 图像压缩",
    9: "第9章 形态学图像处理",
    10: "第10章 图像分割",
    11: "第11章 表示与描述",
    12: "第12章 目标识别",
}

CHAPTER_INTRO = {
    1: "涵盖数字图像处理概念、获取与显示环节模拟（传感器噪声、量化、系统标注）。",
    2: "涵盖采样量化、邻域与连通性、距离度量、卷积及视觉/波段演示。",
    3: "空间域增强核心：灰度变换、直方图处理、平滑/锐化滤波与模糊集增强。",
    4: "频域增强：DFT/FFT 频谱、理想/巴特沃斯/高斯及带通/带阻、同态滤波。",
    5: "退化模型（噪声/模糊）、空域与频域复原、Radon 投影重建。",
    6: "彩色模型转换、彩色增强与滤波、基于颜色的分割。",
    7: "小波分解、金字塔、小波去噪/边缘/压缩演示。",
    8: "无损/有损压缩：霍夫曼、行程、LZW、DCT、JPEG/JPEG2000 及视频帧预测。",
    9: "二值/灰度形态学：腐蚀膨胀、开闭、击中击不中、骨架、顶帽及重建。",
    10: "分割：梯度/Canny 边缘、全局/自适应/Otsu 阈值、区域生长、分水岭。",
    11: "表示与描述：链码、多边形、区域矩、纹理、PCA、Harris/SIFT/ORB。",
    12: "识别：模板匹配、距离分类、贝叶斯、SVM、Hu 矩、BP/CNN 特征演示。",
}


def main():
    os.makedirs(MD_DIR, exist_ok=True)
    by_chapter: dict = {i: [] for i in range(1, 13)}
    alias_rev = {v: k for k, v in ALIASES.items()}

    for name, fn in sorted(METHOD_REGISTRY.items()):
        ch = guess_chapter(name)
        if ch == 0:
            ch = 3
        by_chapter[ch].append((name, fn, alias_rev.get(name, "")))

    for ch, items in by_chapter.items():
        if not items:
            continue
        title = CHAPTER_TITLES[ch]
        path = os.path.join(MD_DIR, f"ch{ch:02d}.md")
        lines = [
            f"# {title}\n",
            f"{CHAPTER_INTRO.get(ch, '')}\n",
            f"本章实现 **{len(items)}** 个图像处理方法，可通过 `main.py` JSON 配置按序调用。\n",
            "## 方法列表\n",
        ]
        for name, fn, alias in items:
            doc = inspect.getdoc(fn) or "无说明"
            lines.append(f"### `{name}`\n")
            if alias:
                lines.append(f"- **中文别名**: {alias}\n")
            lines.append(f"- **说明**: {doc.split(chr(10))[0]}\n")
            lines.append(f"- **JSON 示例**: `{{\"{name}\": \"\"}}` 或 `{{\"{alias or name}\": \"参数1, 参数2\"}}`\n")
            lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"写入 {path}")

    readme = os.path.join(MD_DIR, "README.md")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "# 数字图像处理算法文档\n\n"
            "对应 Gonzalez《数字图像处理》第1–12章。\n\n"
            "## 使用方式\n\n"
            "```bash\n"
            "cd D:\\job_files\\图像算法\\code\n"
            "python main.py -i ..\\img\\test1.png -c configs\\demo_pipeline.json -o ..\\imgout\\result.png\n"
            "```\n\n"
            "## 章节文档\n\n"
        )
        for ch in range(1, 13):
            f.write(f"- [{CHAPTER_TITLES[ch]}](ch{ch:02d}.md)\n")
    print("完成文档生成")


if __name__ == "__main__":
    main()
