"""TCN 训练第 3 步：把「已标注的轨迹 JSON + 原始视频」转成特征序列张量。

输出 ``train.pt``：
    {
      "X":      FloatTensor (N, feature_dim, window)   # 每条滑窗样本一个特征序列
      "y":      LongTensor  (N,)                        # 0=正常 1=溺水
      "C":      FloatTensor (N, 1)                      # 该窗口内 YOLO 平均置信度
      "window": int,
      "feature_dim": int,
      "meta":   [ {"video_name":..., "track_id":..., "start_frame":..., "label":...}, ... ]
    }

X 的维度顺序与 ``model_def.DrowningTCNModel.forward`` 的输入约定一致
（(B, feature_dim, T)，T 为时间维）；C 对应门控融合的 yolo_conf 输入，
因此 ``train_tcn.py`` 与推理端可以共用同一套模型结构。

用法：
    python extract_features.py --tracks data/tracks/reservoir_drowning \
        --video-root D:/videos --out weights/tcn/train.pt
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch

# 允许从仓库任意位置以脚本方式运行
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_def import DrowningTCNModel  # noqa: E402

VIDEO_SUFFIXES = ("", ".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")


def load_labeled_tracks(track_paths):
    """读取若干轨迹 JSON，产出 (track_dict, path) 列表，仅保留已标注（label 0/1）的轨迹。"""
    picked = []
    for path in track_paths:
        try:
            tracks = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 无法解析 {path}: {exc}")
            continue
        if isinstance(tracks, dict):  # 兼容单条轨迹写成 dict 的情况
            tracks = [tracks]
        for trk in tracks or []:
            label = trk.get("label")
            if label not in (0, 1):
                print(f"[INFO] 跳过未标注轨迹 {Path(path).name} track_id={trk.get('track_id')} label={label}")
                continue
            if not trk.get("frames") or not trk.get("bboxes"):
                print(f"[INFO] 跳过空轨迹 {Path(path).name} track_id={trk.get('track_id')}")
                continue
            if len(trk["frames"]) != len(trk["bboxes"]):
                print(f"[WARN] {Path(path).name} frames/bboxes 长度不一致，已跳过")
                continue
            picked.append((trk, Path(path)))
    return picked


def resolve_video(video_name, video_root):
    """在 video_root 下按文件名（含无扩展名）查找轨迹对应的视频。"""
    if video_root is None:
        return None
    stem = Path(video_name).stem
    for cand in (Path(video_name).name, stem):
        for suffix in VIDEO_SUFFIXES:
            if not cand.endswith(suffix) and suffix:
                p = video_root / (cand + suffix)
            else:
                p = video_root / cand
            if p.exists():
                return str(p)
    hits = sorted(p for p in Path(video_root).rglob(f"{stem}*") if p.is_file())
    return str(hits[0]) if hits else None


class SequentialFrameReader:
    """尽量顺序解码的取帧器。

    随机 ``CAP_PROP_POS_FRAMES`` 定位在部分编码（B 帧较多）上会错位，
    这里对「递增」的帧号只做顺序读取，只有需要回退时才 seek。
    """

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频 {video_path}")
        self.next_idx = 0

    def get(self, frame_idx):
        if frame_idx < self.next_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.next_idx = frame_idx
        while self.next_idx < frame_idx:
            if not self.cap.grab():
                return None
            self.next_idx += 1
        ret, frame = self.cap.read()
        self.next_idx += 1
        return frame if ret else None

    def release(self):
        self.cap.release()


def build_windows(feats, confs, window, stride):
    """把一条轨迹的特征序列切成滑窗，返回 (窗口列表, 起始序号列表, 窗口平均置信度列表)。

    帧数不足 window 时，用首帧特征向前补齐，保证仍产出一个窗口。
    """
    if len(feats) < window:
        pad = window - len(feats)
        feats = [feats[0]] * pad + list(feats)
        confs = [confs[0]] * pad + list(confs)
        return [torch.stack(feats, dim=1)], [0], [float(sum(confs) / len(confs))]
    starts = list(range(0, len(feats) - window + 1, stride))
    wins = [torch.stack(feats[s:s + window], dim=1) for s in starts]
    means = [float(sum(confs[s:s + window]) / window) for s in starts]
    return wins, starts, means


def extract(tracks_dir_or_files, video_root, out_path, window=16, stride=None,
            device="auto", tcn_channels=(64, 64, 64, 64)):
    """主流程：轨迹 → 逐帧 bbox 特征 → 滑窗序列 → train.pt。"""
    stride = stride or max(1, window // 2)

    if isinstance(tracks_dir_or_files, (str, Path)) and Path(tracks_dir_or_files).is_dir():
        paths = sorted(Path(tracks_dir_or_files).rglob("*.json"))
    else:
        paths = [Path(p) for p in tracks_dir_or_files]
    if not paths:
        raise SystemExit(f"未找到任何轨迹 JSON：{tracks_dir_or_files}")

    tracks = load_labeled_tracks(paths)
    if not tracks:
        raise SystemExit("没有任何带 label∈{0,1} 的轨迹，请先用 label_tracks.py 标注")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # 只借用特征提取分支（MobileNetV2 backbone + pooling），不需要 TCN 权重
    extractor = DrowningTCNModel(list(tcn_channels), freeze_feature_extractor=True)
    extractor.to(dev)
    extractor.eval()

    xs, cs, ys, metas = [], [], [], []
    cache_path, cache_feats = None, {}     # 同一视频的多条轨迹共享已抽好的帧特征

    for trk, src_path in tracks:
        video_name = trk.get("video_name") or ""
        video_path = resolve_video(video_name, video_root)
        if video_path is None:
            print(f"[WARN] 找不到视频 {video_name!r}（轨迹 {src_path.name}）。"
                  f"请用 --video-root 指定视频目录，本条已跳过")
            continue
        if video_path != cache_path:
            cache_path, cache_feats = video_path, {}

        frame_ids = [int(f) for f in trk["frames"]]
        bboxes = trk["bboxes"]
        trk_confs = trk.get("confs") or [1.0] * len(frame_ids)
        order = sorted(range(len(frame_ids)), key=lambda i: frame_ids[i])

        feats, confs, kept = [], [], []
        reader = SequentialFrameReader(video_path)
        try:
            for i in order:
                fid = frame_ids[i]
                if fid in cache_feats:
                    feat = cache_feats[fid]
                else:
                    frame = reader.get(fid)
                    if frame is None:
                        continue
                    feat = extractor.extract_features_from_bbox(frame, bboxes[i]).detach().cpu()
                    cache_feats[fid] = feat
                if float(feat.sum()) == 0.0:          # 裁剪区域退化，丢弃该帧
                    continue
                feats.append(feat)
                confs.append(float(trk_confs[i]))
                kept.append(i)
        finally:
            reader.release()

        if not feats:
            print(f"[WARN] 轨迹 {src_path.name} track_id={trk.get('track_id')} 没有可用特征")
            continue

        windows, starts, win_confs = build_windows(feats, confs, window, stride)
        for win, conf, st in zip(windows, win_confs, starts):
            xs.append(win)
            cs.append(conf)
            ys.append(int(trk["label"]))
            metas.append({
                "video_name": video_name,
                "track_id": trk.get("track_id"),
                "start_frame": frame_ids[kept[st]] if st < len(kept) else None,
                "source_json": src_path.name,
                "label": int(trk["label"]),
            })
        print(f"[OK] {src_path.name} track_id={trk.get('track_id')} label={trk['label']} "
              f"特征帧 {len(feats)} → 窗口 {len(windows)}")

    if not xs:
        raise SystemExit("没有提取到任何样本（请检查视频路径与标注是否匹配）")

    X = torch.stack(xs)                                # (N, feature_dim, window)
    y = torch.tensor(ys, dtype=torch.long)
    C = torch.tensor(cs, dtype=torch.float32).view(-1, 1)   # (N, 1) 门控融合用的 YOLO 置信度
    payload = {
        "X": X,
        "y": y,
        "C": C,
        "window": window,
        "stride": stride,
        "feature_dim": int(X.shape[1]),
        "tcn_channels": list(tcn_channels),
        "meta": metas,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    print(f"\n已保存 {out_path}\n  X {tuple(X.shape)}  样本 {X.shape[0]}（正常 {n0} / 溺水 {n1}）")
    if 0 in (n0, n1):
        print("  [WARN] 某一类样本数为 0，TCN 学不到该类别，请补充另一类标注后再训练。")


def main():
    root = Path(__file__).resolve().parents[3]         # 项目根目录
    parser = argparse.ArgumentParser(description="轨迹 JSON + 视频 → TCN 特征序列 train.pt")
    parser.add_argument("--tracks", default=str(root / "data" / "tracks"),
                        help="轨迹 JSON 目录或单个文件（默认 data/tracks）")
    parser.add_argument("--video-root", default=None,
                        help="存放轨迹原始视频的目录（按 video_name 匹配）")
    parser.add_argument("--out", default=str(root / "weights" / "tcn" / "train.pt"),
                        help="输出 .pt 路径（默认 weights/tcn/train.pt）")
    parser.add_argument("--window", type=int, default=16, help="滑窗长度，需与推理端一致")
    parser.add_argument("--stride", type=int, default=None, help="滑窗步长，默认 window//2")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    extract(args.tracks, Path(args.video_root) if args.video_root else None,
            args.out, window=args.window, stride=args.stride, device=args.device)


if __name__ == "__main__":
    main()
