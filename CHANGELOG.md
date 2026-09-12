# Changelog

本仓库是对内部工作目录 `drone_detection/` 的**可发布快照**。
为了「别人 clone 下来能跑通」，快照在保留原有算法思路的前提下修复了若干缺陷。
所有改动逐条记录如下；算法结构本身（YOLO + MobileNetV2 + 多尺度 TCN + 图交互 + 门控融合）未做重新设计。

验证方式：14 项冒烟测试全部通过（见 `README.md` 的「自检与验证」一节），
其中包含「修复前会丢标注 / 修复后保留标注」的对照断言。

---

## v1.0.0-snapshot

### 打包与合规

- 剔除 `drone_rescue_env/`（虚拟环境，约 11 GB / 21 万文件）、`data/` 中的图像与视频
  （约 6 GB）、`weights/` 与 `outputs/` 中的 `.pt` 权重（约 570 MB），
  仅保留源码、配置、示例轨迹标注与训练指标记录。仓库体积从 11.8 GB 降到约 6 MB。
- 新增 `LICENSE`：**AGPL-3.0**（与依赖 `ultralytics` 保持一致）。
- 新增 `.gitignore`（按扩展名排除权重/媒体，显式保留 `results.csv` / `results.png` / `args.yaml`
  作为实验记录）与 `.gitattributes`（LF 归一、`.bat` 保持 CRLF、二进制标记、语言统计排除）。
- 新增 `README.md`（本快照的完整使用说明）、`CHANGELOG.md`、`data/README.md`、`docs/`。
- 清理 `outputs/runs/detect/*`：只保留 `results.csv` / `results.png` / `args.yaml`，
  **不发布** `train_batch*.jpg`、`val_batch*.jpg`、`labels.jpg`（这些图块包含数据集中的真实落水画面）
  以及任何 `.pt` 权重。
- 对 `args.yaml` 做了脱敏：把 `D:\drone_detection\drone_rescue_env\Training\...` 这类
  本机绝对路径改写为中性的 `outputs/runs/detect/<run>/weights/best.pt`。
- 目录改名：`data/tracks/Track（水库溺水）/` → `data/tracks/reservoir_drowning/`
  （非 ASCII + 全角括号会在 shell / CI / 部分构建工具里出问题）。
- `docs/img.html` → `docs/algorithm_pseudocode.txt`：该文件不含任何 HTML 标记，
  用浏览器打开只是纯文本，改名后语义才正确。
- `docs/dataset_layout_notes.txt` → `docs/dataset_layout.md`：原文件是 GBK 内容被按 UTF-8
  读取产生的乱码（框线字符全毁），已按原意重写。

### 崩溃级缺陷修复

| 文件 | 问题 | 后果 |
| --- | --- | --- |
| `src/training/tcn/generate_tracks.py` | 残留调试块 `output_data.append({...})`，而 `output_data` 默认是 `None`；随后又把 `output_data` 重新赋值覆盖 | 只要有任何一条 ≥16 帧的轨迹就 `AttributeError` 崩溃；插值结果被丢弃 |
| `src/detection/main.py` | 汇总里读 `detection_stats['total_d']`（不存在的键） | 结束时报 `KeyError`，报告写不出来 |
| `src/detection/main.py` | `with open(json, 'w')` 把 `json` **模块**当文件名 | 报 `TypeError`，JSON 报告永远写不出来 |
| `src/detection/main.py` | README 声称输出 JSON/CSV，代码里只有 JSON 分支且无 CSV 实现 | 按历史产物格式补齐 CSV（`报告摘要` + `检测历史` 两段，`utf-8-sig` 便于 Excel 打开） |
| `src/detection/test.py` | `_send_api_alert` 被写在模块顶层（位于 `if __name__ == "__main__"` 之后），却以 `self._send_api_alert(...)` 调用 | 检测到溺水时立刻 `AttributeError` |
| `src/training/tcn/model_def.py` | 启用 `--flow` 时把 2 通道光流 `cat` 到 3 通道 RGB 上，喂给只接受 3 输入的 MobileNetV2 | `--flow` 必然崩溃；改为光流单独编码→同主干→与 RGB 特征取均值，`feature_dim` 保持不变 |
| `src/training/tcn/extract_features.py` | 整个脚本是半成品；且对 `dict` 调用 `frames.insert(...)` | 无法运行；补全为「轨迹 JSON + 视频 → `train.pt`」的可执行实现 |

### 静默数据损坏（最严重）

- **`src/training/yolo/custom_augs.py` 的 `WaterRipple` 会丢掉全部标注。**
  它按「像素坐标 + `rows=` / `cols=` 关键字」实现 `apply_to_bboxes`，
  但 albumentations 2.x 实际传入的是**归一化 xyxy** 和 `params["shape"]=(H,W,C)`。
  于是 `rows`/`cols` 恒为 0，`np.clip(v, 0, cols - 1)` 变成 `clip(v, 0, -1)`，
  所有框退化成零宽度被 `valid` 过滤掉——**凡是 WaterRipple 命中的增强图，标签全空**，
  等价于把溺水目标当背景训练。
  现已按 2.x 契约重写（并同步实现 `apply_to_keypoints`）；
  取不到 `shape` 时改为**原样返回**，绝不再静默丢标注。
  冒烟测试同时断言「旧实现剩 0 个框」与「新实现剩 2 个框」，锁死这条回归。

### 训练 / 推理一致性

- `train_tcn.py` 原先训练的是一个只有推理端结构子集的 `SimpleTCN`，
  而 `model_def.py` 用 `DrowningTCNModel` 以 `strict=False` 加载它——
  键名对不上时**不会报错，只会静默保留随机初始化权重**。
  现在：`train_tcn.py` 直接训练 `DrowningTCNModel`（主干冻结），
  checkpoint 内写入 `tcn_channels / feature_dim / window_size / use_graph / use_gated_fusion / dropout`，
  推理端先读这些结构参数再建模型，并把 missing/unexpected 键显式打日志。
- `extract_features.py` 现在额外导出 `C`（窗口内 YOLO 平均置信度），
  使门控融合头在离线训练时拿到的输入与推理端一致（缺 `C` 时退回常数 0.5 并打印提示）。
- 图交互分支（`GraphInteraction`）在离线单目标窗口上没有监督信号（它建模的是**同帧多目标**关系），
  因此训练默认关闭并写入 checkpoint，推理端按 checkpoint 记录决定是否启用，
  不再用随机初始化的图网络去污染特征。

### 可移植性

- `src/training/yolo/data.yaml`：`path` 从 `D:\drone_detection\data\drowing(DST1005)`
  改为相对项目根目录的 `data/drowning-DST1005`，支持 `DRONE_DATA_ROOT` 环境变量覆盖；
  缺目录时给出中文可读报错。
  （实测 ultralytics 8.4.61 对相对 `path` 是按它自己的 `DATASETS_DIR` 解析的，
  因此 `train.py` 会先生成一份「path 已解析为绝对路径」的 yaml 副本再交给它。）
- `train.py` / `offline_augment.py` / `extract_features.py` / `train_tcn.py`：
  路径全部基于脚本位置解析，并补齐 argparse（权重、数据集、实验名、epochs、batch、device 等），
  不再依赖当前工作目录。
- `scripts/run_label_studio.bat`：去掉 `D:\Compressed Version\images` 与
  `D:\drone_detection\drone_rescue_env\...` 两处硬编码，改用 `%~dp0` 与
  `LABEL_STUDIO_DOC_ROOT` / `LABEL_STUDIO_PYTHON` 环境变量。
- `detection/test.py`：补 `--api-url` / `--enable-api` 参数
  （原来类支持、命令行无法开启）。
- `generate_tracks.py`：`--min-frames`、`--person-cls` 参数化，输出目录自动创建。

### 兼容性与安全

- **Python 3.14 下 `torch.jit.script` 完全不可用**（连 `torch.nn.Linear` 单独脚本化都会
  `AttributeError: 'Linear' object has no attribute '__annotations__'`）：
  PEP 649 不再为实例暴露 `__annotations__`，而 torch 2.9 的
  `AttributeTypeIsSupportedChecker` 会直接读它。新增 `model_def.ensure_jit_scriptable()`
  在脚本化前给模块子树补空注解字典，实测修补 62 个实例后 TorchScript 路径恢复，
  且输出与 eager 一致。
- `models.mobilenet_v2(pretrained=True)` → `weights=MobileNet_V2_Weights.DEFAULT`
  （前者在 torchvision 0.24 已弃用）。
- TCN 权重加载改为优先 `torch.load(..., weights_only=True)`，避免任意 pickle 执行。
- `detection/test.py` 的告警判定从 `class_name == 'drowning'` 改为大小写不敏感——
  数据集类别名是 `Drowning`，原来的严格比较会让**云端告警永远不触发**。
- `exam.py`：去掉 `🎉` / `❌` 两个 emoji（中文 Windows 下 stdout 重定向到管道时编码退化为
  cp936，会直接抛 `UnicodeEncodeError`），并扩展为覆盖 torch / ultralytics / opencv /
  albumentations≥2 / TorchScript 可用性 / 数据集存在性的完整自检。
- `detection/main.py` / `detection/test.py`：`--imgsz` 原先被 argparse 解析后直接丢弃
  （推理始终用模型默认尺寸），现已接进 `self.model(frame, conf=..., imgsz=...)`。
- `main.py` / `test.py`：`logging.getLogger` 加 handler 前判空，避免重复实例化时日志成倍刷屏。
- `requirements.txt`：按实测通过版本补齐并加约束——`albumentations>=2.0,<3`
  （`WaterRipple` 依赖 2.x 接口）、`ultralytics>=8.3,<9`，新增 `pillow`、`pyyaml`，
  `label-studio` 改为注释并说明必须用 Python ≤3.12。

### 未修改（保留原样，已在 README「已知限制」中说明）

- `src/detection/main.py` 与 `src/detection/test.py` 是**只含 YOLO 的单帧检测器**，
  与 `src/training/tcn/model_def.py` 的 YOLO+TCN 检测器是两套并行实现，同名类不合并，
  以免改变既有调用方式。
- TCN 推理链路仍按 YOLO 类别 id `0` 作为跟踪目标（COCO `person`）。
  若换成 3 类的溺水模型，`0` 是 `Swimming`，`Drowning`/`Person out of water` 不会被跟踪。
- `SimpleTracker` 仍是「卡尔曼 + 贪心 IoU」，无外观特征，遮挡后 ID 切换未解决。
- 随仓库发布的 4 条示例轨迹**全部是正样本（label=1）**，缺少负样本，
  不足以复现 TCN 训练，需要自行补标注。
