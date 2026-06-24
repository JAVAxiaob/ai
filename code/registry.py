# -*- coding: utf-8 -*-
"""全局方法注册表：方法名 -> 处理函数。"""

from __future__ import annotations

import inspect
from typing import Callable, Dict

from ch01_intro import methods as ch01
from ch02_basics import methods as ch02
from ch03_spatial import methods as ch03
from ch04_frequency import methods as ch04
from ch05_restoration import methods as ch05
from ch06_color import methods as ch06
from ch07_wavelet import methods as ch07
from ch08_compression import methods as ch08
from ch09_morphology import methods as ch09
from ch10_segmentation import methods as ch10
from ch11_representation import methods as ch11
from ch12_recognition import methods as ch12

MethodFunc = Callable

# 中文别名映射到英文函数名
ALIASES: Dict[str, str] = {
    "应用领域标注": "application_domains_demo",
    "模拟传感器噪声": "simulate_sensor_noise",
    "量化模拟": "simulate_quantization",
    "系统信息标注": "annotate_system_info",
    "显示缩放": "resize_for_display",  "降采样": "downsample",
    "灰度量化": "quantize_levels",
    "邻域标记": "mark_neighbors",
    "连通分量": "connectivity_components_vis",
    "视觉错觉": "optical_illusion_demo",
    "概率噪声演示": "probability_noise_demo",
    "波段成像": "spectrum_band_simulation",
    "距离变换可视化": "distance_transform_vis",
    "卷积演示": "convolve_demo",
    "添加高斯噪声": "add_gaussian_noise",
    "线性变换": "linear_transform",
    "对比度拉伸": "contrast_stretch",
    "对数变换": "log_transform",
    "幂律变换": "power_law_transform",
    "全局阈值": "threshold_global",
    "直方图均衡化": "histogram_equalization",
    "直方图规定化": "histogram_matching",
    "局部直方图均衡": "local_histogram_equalization",
    "均值滤波": "mean_filter",
    "高斯滤波": "gaussian_filter",
    "中值滤波": "median_filter",
    "Sobel梯度": "sobel_gradient",
    "拉普拉斯锐化": "laplacian_sharpen",
    "高通滤波": "highpass_filter",
    "非锐化掩模": "unsharp_mask",
    "模糊集增强": "fuzzy_enhancement",
    "频谱显示": "dft_spectrum",
    "理想低通": "ideal_lowpass",
    "理想高通": "ideal_highpass",
    "巴特沃斯低通": "butterworth_lowpass",
    "巴特沃斯高通": "butterworth_highpass",
    "高斯低通": "gaussian_lowpass",
    "高斯高通": "gaussian_highpass",
    "带通滤波": "bandpass_filter",
    "带阻滤波": "bandreject_filter",
    "同态滤波": "homomorphic_filter",
    "FFT滤波": "fft_filter_pipeline",
    "椒盐噪声": "add_salt_pepper_noise",
    "周期噪声": "add_periodic_noise",
    "运动模糊": "motion_blur",
    "散焦模糊": "defocus_blur",
    "均值去噪": "mean_denoise",
    "中值去噪": "median_denoise",
    "自适应去噪": "adaptive_denoise",
    "维纳滤波": "wiener_filter_spatial",
    "逆滤波": "inverse_filter",
    "频域维纳": "wiener_filter_freq",
    "约束最小二乘": "constrained_least_squares",
    "盲复原": "blind_deconvolution",
    "Radon重建": "radon_reconstruction_demo",
    "RGB转HSV": "rgb_to_hsv",
    "RGB转HSI": "rgb_to_hsi",
    "RGB转CMY": "rgb_to_cmy",
    "RGB转CMYK": "rgb_to_cmyk",
    "RGB转YCbCr": "rgb_to_ycbcr",
    "彩色转灰度": "color_to_gray",
    "彩色对比度增强": "color_contrast_enhance",
    "伪彩色": "pseudocolor",
    "彩色平滑": "color_smooth",
    "彩色锐化": "color_sharpen",
    "彩色边缘": "color_edge_canny",
    "颜色阈值分割": "color_threshold_segment",
    "彩色区域生长": "color_region_growing",
    "小波分解": "dwt_decompose",
    "金字塔降采样": "pyramid_downsample",
    "小波去噪": "wavelet_denoise",
    "小波边缘": "wavelet_edge_detect",
    "小波压缩": "wavelet_compress_demo",
    "算术编码": "arithmetic_encode_vis",
    "视频帧预测": "video_frame_prediction_demo",
    "霍夫曼演示": "huffman_encode_decode_vis",
    "行程编码": "run_length_encode_vis",
    "LZW压缩": "lzw_compress_vis",
    "DCT压缩": "dct_compress",
    "JPEG压缩": "jpeg_quality_compress",
    "JPEG2000压缩": "jpeg2000_compress",
    "腐蚀": "erosion",
    "膨胀": "dilation",
    "开运算": "opening",
    "闭运算": "closing",
    "击中击不中": "hit_or_miss",
    "边界提取": "boundary_extract",
    "骨架提取": "skeleton_extract",
    "细化": "thinning",
    "粗化": "thickening",
    "距离变换": "distance_transform",
    "灰度腐蚀": "grayscale_erosion",
    "灰度膨胀": "grayscale_dilation",
    "灰度开运算": "grayscale_opening",
    "灰度闭运算": "grayscale_closing",
    "顶帽": "top_hat",
    "黑帽": "black_hat",
    "形态学梯度": "morphological_gradient",
    "形态学重建": "morphological_reconstruction",
    "Sobel边缘": "sobel_edge",
    "Prewitt边缘": "prewitt_edge",
    "拉普拉斯边缘": "laplacian_edge",
    "Canny边缘": "canny_edge",
    "Otsu阈值": "otsu_threshold",
    "自适应阈值": "adaptive_threshold",
    "多阈值": "multi_threshold",
    "区域生长": "region_growing",
    "区域分裂合并": "region_split_merge",
    "分水岭": "watershed_segment",
    "边缘区域结合": "combined_edge_region",
    "标记图边界": "marked_graph_boundary",
    "链码边界": "chain_code_boundary",
    "多边形近似": "polygon_approximation",
    "区域矩特征": "region_moments_vis",
    "纹理方差": "texture_glcm_vis",
    "纹理频谱": "texture_spectral_vis",
    "纹理结构": "texture_structural_vis",
    "PCA重建": "pca_projection",
    "Harris角点": "harris_corners",
    "SIFT特征": "sift_features",
    "SURF特征": "surf_features",
    "ORB特征": "orb_features",
    "模板匹配": "template_matching",
    "最小距离分类": "minimum_distance_classifier_demo",
    "贝叶斯阈值": "bayesian_threshold_demo",
    "SVM分割": "svm_segment_demo",
    "Hu矩匹配": "shape_matching_hu",
    "图匹配": "graph_matching_demo",
    "CNN特征演示": "cnn_feature_map_demo",
    "BP神经网络": "bp_neural_demo",
    "句法模式识别": "syntactic_pattern_demo",
}


def _collect(module) -> Dict[str, MethodFunc]:
    """仅注册本模块 methods.py 中定义的函数，排除 import 进来的符号。"""
    out: Dict[str, MethodFunc] = {}
    mod_name = getattr(module, "__name__", "")
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        if getattr(fn, "__module__", None) != mod_name:
            continue
        out[name] = fn
    return out


METHOD_REGISTRY: Dict[str, MethodFunc] = {}
for mod in (ch01, ch02, ch03, ch04, ch05, ch06, ch07, ch08, ch09, ch10, ch11, ch12):
    # _collect(mod) 返回该模块内所有注册函数的字典
    # update 批量并入总注册表
    METHOD_REGISTRY.update(_collect(mod))


def resolve_method(name: str) -> MethodFunc:
    """根据英文名或中文别名解析方法。"""
    key = name.strip()
    if key in METHOD_REGISTRY:
        return METHOD_REGISTRY[key]
    if key in ALIASES:
        return METHOD_REGISTRY[ALIASES[key]]
    raise KeyError(f"未知方法: {name}，可用方法见 registry.METHOD_REGISTRY")
