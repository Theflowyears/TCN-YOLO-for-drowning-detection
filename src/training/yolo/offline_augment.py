"""离线数据增强：对数据集 images/train 生成 images/train_aug + labels/train_aug。

默认作用于 data/drowning-DST1005（与 src/training/yolo/data.yaml 的 train 列表对应），
可用 --data-root 指向任意同结构数据集。产物目录名固定为 train_aug / labels 同名目录，
data.yaml 已经把该目录列入 train  split。

用法：
    python src/training/yolo/offline_augment.py
    python src/training/yolo/offline_augment.py --data-root data/traindata2 --num-aug 2
"""

import argparse
import os
from pathlib import Path

import albumentations as A
import cv2
from tqdm import tqdm

from custom_augs import WaterRipple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                     # 项目根目录
IMAGE_GLOBS = ("*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]", "*.[pP][nN][gG]", "*.[bB][mM][pP]")


def build_transform():
    return A.Compose([
        WaterRipple(p=0.5, amplitude_range=(0.5, 2.0), frequency_range=(0.01, 0.03)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.ISONoise(p=0.2),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))


def read_yolo_labels(label_path):
    """读取 YOLO txt 标注，裁剪到 [0,1] 并丢弃退化框。"""
    boxes, class_labels = [], []
    if not os.path.exists(label_path):
        return boxes, class_labels
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x_c, y_c, w, h = map(float, parts[1:5])

            x_min = max(0.0, min(1.0, x_c - w / 2))
            x_max = max(0.0, min(1.0, x_c + w / 2))
            y_min = max(0.0, min(1.0, y_c - h / 2))
            y_max = max(0.0, min(1.0, y_c + h / 2))
            if x_max - x_min <= 0 or y_max - y_min <= 0:
                continue

            boxes.append([(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min])
            class_labels.append(cls)
    return boxes, class_labels


def find_label(img_path, raw_img_dir, raw_lbl_dir):
    """按「同目录 → labels 对应子目录」的顺序查找标注，兼容大写扩展名。"""
    candidates = [img_path.with_suffix(".txt"), img_path.with_suffix(".TXT")]
    try:
        rel = img_path.relative_to(raw_img_dir)
        candidates += [raw_lbl_dir / rel.with_suffix(".txt"), raw_lbl_dir / rel.with_suffix(".TXT")]
    except ValueError:
        pass
    candidates.append(raw_lbl_dir / img_path.with_suffix(".txt").name)
    candidates.append(raw_lbl_dir / img_path.with_suffix(".TXT").name)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def augment(raw_img_dir, raw_lbl_dir, aug_img_dir, aug_lbl_dir, num_aug=3):
    raw_img_dir, raw_lbl_dir = Path(raw_img_dir), Path(raw_lbl_dir)
    aug_img_dir, aug_lbl_dir = Path(aug_img_dir), Path(aug_lbl_dir)
    if not raw_img_dir.is_dir():
        raise SystemExit(f"图像目录不存在：{raw_img_dir}")

    aug_img_dir.mkdir(parents=True, exist_ok=True)
    aug_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted({p for glob in IMAGE_GLOBS for p in raw_img_dir.rglob(glob)})
    print(f"在 {raw_img_dir} 中发现 {len(image_files)} 张训练图像，每张生成 {num_aug} 份增强...")
    if not image_files:
        raise SystemExit("没有找到任何图像，请检查 --data-root")

    transform = build_transform()
    written, skipped = 0, 0
    for img_path in tqdm(image_files, desc="offline augment"):
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        label_path = find_label(img_path, raw_img_dir, raw_lbl_dir)
        bboxes, class_labels = read_yolo_labels(str(label_path)) if label_path else ([], [])
        if not bboxes:
            print(f"警告：{img_path.name} 没有有效标注，跳过")
            skipped += 1
            continue

        for i in range(num_aug):
            transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            stem, suffix = img_path.stem, img_path.suffix
            aug_name = f"{stem}_aug{i}{suffix}"
            cv2.imwrite(str(aug_img_dir / aug_name), transformed["image"])
            # 用 with_suffix 而不是 str.replace，避免文件名本身含扩展名片段时改错
            aug_label_path = aug_lbl_dir / Path(aug_name).with_suffix(".txt")
            with open(str(aug_label_path), "w", encoding="utf-8") as f:
                for bbox, cls in zip(transformed["bboxes"], transformed["class_labels"]):
                    f.write(f"{int(cls)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            written += 1

    print(f"离线增强完成！新增图像 {written} 张 → {aug_img_dir}（跳过 {skipped} 张）")


def main():
    parser = argparse.ArgumentParser(description="YOLO 数据集离线增强")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "drowning-DST1005"),
                        help="数据集根目录（默认 data/drowning-DST1005）")
    parser.add_argument("--train-split", default="train", help="源图像子目录名（默认 train）")
    parser.add_argument("--aug-split", default="train_aug", help="输出子目录名（默认 train_aug）")
    parser.add_argument("--num-aug", type=int, default=3, help="每张源图生成的增强图数量")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    augment(data_root / "images" / args.train_split, data_root / "labels" / args.train_split,
            data_root / "images" / args.aug_split, data_root / "labels" / args.aug_split,
            num_aug=args.num_aug)


if __name__ == "__main__":
    main()
