"""YOLO 训练入口。

与快照原版相比的改动：数据集路径不再硬编码为某台机器的绝对路径，
而是按「项目根目录 / data.yaml 的 path（或环境变量 DRONE_DATA_ROOT）」解析，
并写出一份 path 已解析为绝对路径的副本交给 ultralytics
（ultralytics 对相对 path 会按它自己的 DATASETS_DIR 解析，因此必须交给它绝对路径）。

用法：
    python src/training/yolo/train.py                        # 用默认配置
    python src/training/yolo/train.py --weights weights/yolo/yolov8n.pt --epochs 100
    DRONE_DATA_ROOT=/data/drowning-DST1005 python .../train.py
"""

import argparse
import os
import re
from pathlib import Path

import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                     # 项目根目录


def resolve_dataset_root(data_yaml: Path) -> Path:
    """确定数据集根目录：环境变量优先，其次 data.yaml 的 path（相对项目根）。"""
    raw = os.environ.get("DRONE_DATA_ROOT")
    if not raw:
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        raw = cfg.get("path")
    if not raw:
        raise SystemExit(f"{data_yaml} 没有 path 键，且未设置 DRONE_DATA_ROOT")

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        raise SystemExit(
            f"数据集目录不存在：{path}\n"
            f"请准备好数据后修改 {data_yaml} 的 path，或设置环境变量 DRONE_DATA_ROOT")
    return path


def write_resolved_yaml(data_yaml: Path, dataset_root: Path, out_dir: Path, name: str) -> Path:
    """生成 path 为绝对路径的 data.yaml 副本，供 ultralytics 使用。"""
    text = data_yaml.read_text(encoding="utf-8")
    resolved = str(dataset_root).replace("\\", "/")
    if re.search(r"(?m)^\s*path\s*:", text):
        new_text = re.sub(r"(?m)^\s*path\s*:.*$", f"path: {resolved}", text, count=1)
    else:
        new_text = f"path: {resolved}\n" + text
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.data.yaml"
    out.write_text(new_text, encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description="训练溺水检测 YOLO")
    parser.add_argument("--weights", default=str(ROOT / "weights" / "yolo" / "yolov8s.pt"),
                        help="起始权重（默认 weights/yolo/yolov8s.pt）")
    parser.add_argument("--data", default=str(HERE / "data.yaml"),
                        help="数据集配置 yaml（默认与本脚本同目录的 data.yaml）")
    parser.add_argument("--hyp", default=str(HERE / "hyp_strong.yaml"),
                        help="在线增强超参数 yaml（默认 hyp_strong.yaml）")
    parser.add_argument("--project", default=str(ROOT / "outputs" / "runs" / "detect"),
                        help="训练产物根目录（默认 outputs/runs/detect）")
    parser.add_argument("--name", default="dual_aug_exp3", help="实验名称")
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--device", default="0", help="GPU 编号（如 0）或 cpu")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--freeze", type=int, default=10, help="冻结 backbone 前 N 层，0 表示不冻结")
    parser.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    parser.add_argument("--no-cos-lr", action="store_true", help="关闭余弦退火学习率")
    parser.add_argument("--cache", action="store_true", help="把数据集缓存到内存/磁盘以加速")
    args = parser.parse_args()

    data_yaml = Path(args.data).expanduser().resolve()
    dataset_root = resolve_dataset_root(data_yaml)
    resolved_yaml = write_resolved_yaml(data_yaml, dataset_root, Path(args.project) / "_config",
                                        args.name)
    print(f"数据集根目录: {dataset_root}")
    print(f"实际使用的配置副本: {resolved_yaml}")

    model = YOLO(args.weights)

    model.train(
        data=str(resolved_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        amp=not args.no_amp,
        cache=args.cache,
        cfg=args.hyp,
        cos_lr=not args.no_cos_lr,
        patience=args.patience,
        freeze=args.freeze or None,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
