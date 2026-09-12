# 数据集目录规范

原始快照里的 `docs/dataset_layout_notes.txt` 是 GBK 内容被按 UTF-8 读取的乱码文件，
本文件为其可读重写版（见 `CHANGELOG.md`）。

YOLO 训练链路（`src/training/yolo/`）要求数据集按下面的布局放在 `data/<数据集名>/` 下，
即 Ultralytics 标准结构（Roboflow 导出后微调目录名即可）：

```text
drowning_dataset/
├── images/              # 所有图像
│   ├── train/           #   训练集图像
│   ├── train_aug/       #   离线增强图像（由 offline_augment.py 生成）
│   ├── valid/           #   验证集图像（data.yaml 的 val 指向这里）
│   └── test/            #   测试集图像（可选）
├── labels/              # 与图像同名同层级的 YOLO 标注
│   ├── train/
│   ├── train_aug/
│   ├── valid/
│   └── test/
└── data.yaml            # 数据集自带的配置（仓库另有一份等效配置，见下方）
```

要点：

1. `images/<split>/x.jpg` 与 `labels/<split>/x.txt` 必须同名，标注为
   `<cls> <x_center> <y_center> <width> <height>`，坐标按图像宽高归一化到 `[0,1]`。
2. 类别索引从 0 开始，与 `data.yaml` 的 `names` 顺序一致。
3. 本项目使用 3 类：`Swimming` / `Drowning` / `Person out of water`。
   推理端 `src/detection/main.py` 按类别名 `Drowning` 触发告警（大小写不敏感）。
4. 训练脚本读取的是 **`src/training/yolo/data.yaml`**，其 `path` 指向数据集根目录；
   数据集自带的 `data.yaml` 仅作归档，不参与训练。
5. `offline_augment.py` 会把产物写回同一个数据集根目录下的 `images/train_aug`，
   因此增强结果与原始数据一起被 `.gitignore` 排除，不会进入 Git。

## TCN 轨迹数据

TCN 链路（`src/training/tcn/`）不吃图像数据集，而是吃「轨迹 JSON」：

```text
data/tracks/
└── reservoir_drowning/          # 本仓库随包附带的示例标注（仅正样本）
    ├── drowning_person2.json
    ├── drowning_track.json
    ├── drowning_track3.json
    └── drowning_track4.json
```

JSON 为数组，每个元素一条轨迹：

```jsonc
{
  "track_id": 0,
  "video_name": "xxx.mp4",     // extract_features.py 据此在 --video-root 下查找视频
  "frames": [2340, 2341, ...], // 帧序号，递增
  "bboxes": [[1485,590,1591,708], ...],  // 像素坐标 xyxy，与 frames 一一对应
  "confs":  [1.0, 1.0, ...],   // 该帧 YOLO 置信度（手动标注轨迹填 1.0）
  "label":  1                  // 0=正常 1=溺水，-1 或缺失视为未标注
}
```
