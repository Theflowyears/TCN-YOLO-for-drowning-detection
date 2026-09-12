"""环境自检：确认 GPU 与整条流水线依赖是否可用。

原快照里这个脚本只打印 torch/CUDA 信息，且用了 🎉 / ❌ 两个 emoji ——
在中文 Windows 上把输出重定向到管道时（stdout 编码退化为 cp936/GBK），
Python 会直接抛 UnicodeEncodeError 而不是打印结果，所以这里改成 ASCII 标记。

用法：
    python src/training/yolo/exam.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

print(f"Python      : {sys.version.split()[0]}  ({sys.executable})")
print(f"项目根目录  : {ROOT}")

ok = True


def report(name, fn):
    global ok
    try:
        detail = fn()
        print(f"  [OK]   {name:<14} {detail}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] {name:<14} {type(exc).__name__}: {exc}")


def check_torch():
    import torch
    cuda = torch.cuda.is_available()
    info = f"{torch.__version__}  CUDA可用={cuda}"
    if cuda:
        info += f"  设备={torch.cuda.get_device_name(0)}"
    return info


def check_torchvision():
    import torchvision
    return torchvision.__version__


def check_ultralytics():
    import ultralytics
    return ultralytics.__version__


def check_cv2():
    import cv2
    return cv2.__version__


def check_numpy():
    import numpy
    return numpy.__version__


def check_albumentations():
    import albumentations
    v = albumentations.__version() if callable(getattr(albumentations, "__version__", None)) \
        else albumentations.__version__
    major = int(str(v).split(".")[0])
    if major < 2:
        raise RuntimeError(f"需要 >=2.0（custom_augs.WaterRipple 依赖 2.x 的 bbox 接口），当前 {v}")
    return f"{v}  (>=2.0 必需)"


def check_torchscript():
    import torch
    import torch.nn as nn
    try:
        torch.jit.script(nn.Linear(2, 2))
    except AttributeError as exc:
        # Python 3.14 + torch 2.9 的已知不兼容，详见 model_def.ensure_jit_scriptable
        raise RuntimeError("torch.jit.script 不可用（Python 3.14 与 torch 2.9 冲突）；"
                           "推理端请加 --torchscript 前先阅读 README「已知限制」") from exc
    return "可脚本化"


def check_datasets():
    hits = []
    for cand in ("drowning-DST1005", "drowing(DST1005)", "traindata2"):
        d = ROOT / "data" / cand
        if (d / "images").is_dir():
            hits.append(f"data/{cand}")
    if not hits:
        raise FileNotFoundError(
            "未找到数据集目录（data/<数据集>/images/...），见 docs/dataset_layout.md")
    return ", ".join(hits)


print("\n依赖检查：")
report("torch", check_torch)
report("torchvision", check_torchvision)
report("ultralytics", check_ultralytics)
report("opencv", check_cv2)
report("numpy", check_numpy)
report("albumentations", check_albumentations)
report("torchscript", check_torchscript)
report("dataset", check_datasets)

if ok:
    print("\n[OK] 环境检查通过，可以开始训练与推理。")
else:
    print("\n[WARN] 存在未通过项，请按上面的提示补齐后再运行。")
sys.exit(0 if ok else 1)
