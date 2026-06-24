# -*- coding: utf-8 -*-
"""批量测试所有方法并保存结果。"""

import json
import os
import sys
import traceback

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODE_DIR)

from main import process_pipeline
from registry import METHOD_REGISTRY
from utils import load_image, save_image

IMG_DIR = r"..\img"
OUT_DIR = r"..\imgout"
CONFIG_DIR = r"..\configs"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    images = [
        os.path.join(IMG_DIR, "test1.png"),
        os.path.join(IMG_DIR, "test2.png"),
    ]
    failed = []
    ok_count = 0
    for img_path in images:
        if not os.path.isfile(img_path):
            print(f"跳过，图像不存在: {img_path}")
            continue
        base = os.path.splitext(os.path.basename(img_path))[0]
        for method_name in sorted(METHOD_REGISTRY.keys()):
            try:
                out_path = os.path.join(OUT_DIR, f"{base}_{method_name}.png")
                process_pipeline(img_path, {method_name: ""}, out_path)
                ok_count += 1
            except Exception as e:
                failed.append((method_name, img_path, str(e)))
                traceback.print_exc()

    # 流水线 JSON 测试
    pipeline_cfg = os.path.join(CONFIG_DIR, "demo_pipeline.json")
    if os.path.isfile(pipeline_cfg):
        with open(pipeline_cfg, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for img_path in images:
            if os.path.isfile(img_path):
                base = os.path.splitext(os.path.basename(img_path))[0]
                out = os.path.join(OUT_DIR, f"{base}_pipeline_result.png")
                process_pipeline(img_path, cfg, out)
                ok_count += 1

    print(f"\n成功: {ok_count}, 失败: {len(failed)}")
    if failed:
        print("失败列表（前20）:")
        for item in failed[:20]:
            print(item)
    return len(failed) == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
