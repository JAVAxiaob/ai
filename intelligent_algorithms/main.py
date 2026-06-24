# -*- coding: utf-8 -*-
"""机器学习与深度学习教程总入口。

用法:
  python main.py --list
  python main.py -m linear_regression_demo -o ./output
"""
from __future__ import annotations

import argparse
import os
import sys

import inspect

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from registry import METHOD_REGISTRY, ALIASES, resolve
from utils import parse_params, ensure_dir


def run(method_name, params=None, output=None):
    fn = resolve(method_name)
    sig = inspect.signature(fn)
    kwargs = {}
    args = parse_params(params) if params else []
    if output:
        ensure_dir(output)
        if 'output' in sig.parameters:
            kwargs['output'] = output
    return fn(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description='机器学习与深度学习教程')
    parser.add_argument('--method', '-m', help='方法名')
    parser.add_argument('--params', '-p', default='', help='参数，逗号分隔')
    parser.add_argument('--output', '-o', default='./output', help='输出目录')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()

    if args.list or not args.method:
        print('== 可用方法 ==')
        for n in sorted(METHOD_REGISTRY.keys()):
            print('  -', n)
        print('')
        print('== 中文别名 ==')
        for k, v in ALIASES.items():
            print('  %s -> %s' % (k, v))
        return

    ensure_dir(args.output)
    result = run(args.method, args.params, args.output)
    print('done: %s -> %s' % (args.method, result))


if __name__ == '__main__':
    main()
