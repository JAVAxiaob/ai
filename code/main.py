# -*- coding: utf-8 -*-
"""
数字图像处理流水线总入口。

用法:
  python main.py --image 输入图 --config 配置.json --output 输出路径

JSON 格式（按顺序执行）:
{
  "histogram_equalization": "",
  "gaussian_filter": "5, 1.0",
  "Canny边缘": "50, 150"
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Union

# 保证从 code 目录导入
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from registry import METHOD_REGISTRY, resolve_method
from utils import ImageArray, load_image, parse_params, save_image


def process_pipeline(
    image: Union[str, ImageArray],
    pipeline_config: Dict[str, Any],
    output_path: str | None = None,
) -> ImageArray:
    """
    对图像按 JSON 配置依次应用处理方法。

    :param image: 图像路径或 ndarray
    :param pipeline_config: {方法名: 参数字符串或列表}
    :param output_path: 可选，保存最终结果
    :return: 处理后的图像
    """
    # isinstance(image, str) 精准判断输入是不是文件路径字符串
    if isinstance(image, str):
        img = load_image(image)
    else:
        img = image.copy()

    for method_name, param_value in pipeline_config.items():
        if method_name in ("pipeline", "comment", "说明"):
            continue
        if isinstance(param_value, dict):
            # 支持 {"method": "x", "params": "..."} 列表形式的外层
            continue
        fn = resolve_method(str(method_name))
        params = parse_params(param_value)
        print(params)
        img = fn(img, *params)

    if output_path:
        save_image(output_path, img)
    return img


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 JSON 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "pipeline" in data and isinstance(data["pipeline"], list):
        ordered = {}
        for item in data["pipeline"]:
            name = item.get("method") or item.get("name")
            params = item.get("params", "")
            ordered[name] = params
        return ordered
    return {k: v for k, v in data.items() if not k.startswith("_")}


def run_from_paths(
    image_path: str,
    config_path: str,
    output_path: str,
) -> ImageArray:
    """从文件路径运行完整流水线。"""
    config = load_config(config_path)
    return process_pipeline(image_path, config, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="数字图像处理流水线")
    parser.add_argument("--image", "-i", required=True, help="输入图像路径")
    parser.add_argument("--config", "-c", required=True, help="JSON 配置文件")
    parser.add_argument("--output", "-o", required=True, help="输出图像路径")
    parser.add_argument("--list", action="store_true", help="列出所有可用方法")
    args = parser.parse_args()

    if args.list:
        for name in sorted(METHOD_REGISTRY.keys()):
            print(name)
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result = run_from_paths(args.image, args.config, args.output)
    print(f"完成: {args.output}, shape={result.shape}")


if __name__ == "__main__":
    main()
