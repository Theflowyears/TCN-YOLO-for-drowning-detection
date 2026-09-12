# data/

这个目录下的图像、视频与增强产物都**不进入 Git**（见根目录 `.gitignore`），
只随仓库附带 TCN 链路所需的示例轨迹标注。

放置方式：

```text
data/
├── drowning-DST1005/      # YOLO 图像数据集，布局见 docs/dataset_layout.md
│   ├── images/{train,train_aug,valid,test}/
│   └── labels/{train,train_aug,valid,test}/
├── tracks/                # ✅ 已随仓库提供：示例轨迹标注（仅正样本）
│   └── reservoir_drowning/*.json
└── raw_images/            # Label Studio 的本地图片根目录（可选）
```

`data/drowning-DST1005` 只是默认目录名，可用 `src/training/yolo/data.yaml`
的 `path` 键或环境变量 `DRONE_DATA_ROOT` 指向任意位置。

数据集本身来自公开的溺水检测标注集（Roboflow 生态），本仓库不重新分发图像；
请按你实际使用的数据集许可证自行获取。
