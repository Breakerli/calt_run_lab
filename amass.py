"""下载并保存 youtopia/AMASS 动作数据集（ModelScope）。

AMASS 数据集中部分字段为 object 类型 numpy 数组，datasets 生成 split 时内部使用
``np.load(..., allow_pickle=False)`` 读取，会导致 ``ValueError: Object arrays cannot be
loaded when allow_pickle=False``。本脚本在加载前将 ``numpy.load`` 默认行为临时改为
``allow_pickle=True``，加载成功后保存为本地 Arrow 格式，后续无需再次 patch。

运行示例:
    python amass.py
    python amass.py --output ./amass_data --split train --overwrite
"""

import argparse
import os
import shutil
import sys

import numpy as np
import numpy.lib.format as _np_format
from modelscope.msdatasets import MsDataset


# --- Numpy allow_pickle 补丁 ---
# datasets 生成 AMASS split 时读取 .npz/.npy，其中包含 object 数组，默认
# allow_pickle=False 会抛 ValueError。我们在 numpy 最底层 format.read_array
# 强制 allow_pickle=True，覆盖所有调用路径（np.load、NpzFile、直接调用等）。
_ORIG_READ_ARRAY = _np_format.read_array


def _read_array_allow_pickle(fp, allow_pickle=False, pickle_kwargs=None, *, max_header_size=10000):
    return _ORIG_READ_ARRAY(fp, allow_pickle=True, pickle_kwargs=pickle_kwargs, max_header_size=max_header_size)


_np_format.read_array = _read_array_allow_pickle

_ORIG_NP_LOAD = np.load


def _np_load_allow_pickle(*args, **kwargs):
    kwargs.setdefault("allow_pickle", True)
    return _ORIG_NP_LOAD(*args, **kwargs)


np.load = _np_load_allow_pickle


# --- Constants ---
DEFAULT_OUTPUT = "./amass_data"
DEFAULT_SUBSET = None
DEFAULT_SPLIT = "train"


# --- Utilities ---
def parse_args():
    parser = argparse.ArgumentParser(description="下载并保存 AMASS 数据集")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="保存目录（默认 ./amass_data）")
    parser.add_argument("--subset", type=str, default=DEFAULT_SUBSET, help="子集名（可选）")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, help="split 名称，默认 train")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出目录")
    return parser.parse_args()


def save_dataset(ds, output_dir, split_name):
    """将数据集保存为本地 Arrow 格式，并额外保存列信息摘要。"""
    os.makedirs(output_dir, exist_ok=True)
    split_dir = os.path.join(output_dir, split_name)

    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)

    # MsDataset 底层通常是 datasets.Dataset，使用 save_to_disk 效率最高；
    # 若不是，则回退到按列保存 .npy。
    if hasattr(ds, "save_to_disk"):
        ds.save_to_disk(split_dir)
        print(f"已保存 {len(ds)} 条样本到 {split_dir}")
    else:
        os.makedirs(split_dir, exist_ok=True)
        for col in ds.column_names:
            data = ds[col]
            try:
                arr = np.array(data)
            except ValueError:
                arr = np.array(data, dtype=object)
            np.save(os.path.join(split_dir, f"{col}.npy"), arr)
        print(f"已保存 {len(ds)} 条样本到 {split_dir}（按列 .npy）")

    summary_path = os.path.join(output_dir, f"{split_name}_columns.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"num_examples: {len(ds)}\n")
        f.write(f"columns: {ds.column_names}\n")
    print(f"列信息摘要：{summary_path}")


def main():
    args = parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        print(f"输出目录已存在：{args.output}，请使用 --overwrite 覆盖或更换 --output")
        sys.exit(1)

    load_kwargs = {"split": args.split}
    if args.subset:
        load_kwargs["subset_name"] = args.subset

    print(f"正在加载 youtopia/AMASS，split={args.split} ...")
    ds = MsDataset.load("youtopia/AMASS", **load_kwargs)
    print(f"加载成功：{len(ds)} 条样本，字段：{ds.column_names}")

    save_dataset(ds, args.output, args.split)
    print(f"完成：数据已保存到 {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
