"""TCN 训练第 4 步：在离线特征序列上训练溺水分类器。

与推理端 ``model_def.DrowningDetector`` 保持同一套模型结构：
直接实例化 ``DrowningTCNModel``（MobileNetV2 主干冻结），只训练
TCN / 分类头 / 门控融合，因此产出的 ``.pth`` 可被推理端原样加载。

产出的 checkpoint 会把结构超参一起写进去（tcn_channels / feature_dim /
window_size / use_graph / use_gated_fusion），推理端读取后按同样结构重建模型，
避免「训练一套、推理另一套」导致权重静默失配。

用法：
    python train_tcn.py --data weights/tcn/train.pt --out weights/tcn/tcn_drowning.pth
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_def import DrowningTCNModel  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_payload(path):
    """读取 extract_features.py 的产物，返回 (X, y, C, cfg)。"""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "X" not in blob or "y" not in blob:
        raise SystemExit(f"{path} 不是 extract_features.py 产出的特征文件（需要含 X/y 的 dict）")
    X = blob["X"].float()
    y = blob["y"].long()
    C = blob.get("C")
    if C is None:                                    # 兼容旧版：无置信度时用 0.5 占位
        C = torch.full((X.shape[0], 1), 0.5, dtype=torch.float32)
        print("[INFO] 特征文件缺少 C（YOLO 置信度），门控融合将使用常数 0.5")
    else:
        C = C.float().view(-1, 1)
    cfg = {
        "tcn_channels": list(blob.get("tcn_channels", [64, 64, 64, 64])),
        "feature_dim": int(blob.get("feature_dim", X.shape[1])),
        "window_size": int(blob.get("window", X.shape[2])),
    }
    if X.shape[0] != y.shape[0] or X.shape[0] != C.shape[0]:
        raise SystemExit(f"样本数不一致：X {X.shape[0]} / y {y.shape[0]} / C {C.shape[0]}")
    if X.shape[1] != cfg["feature_dim"]:
        raise SystemExit(f"feature_dim 与 X 通道数不符：{cfg['feature_dim']} vs {X.shape[1]}")
    return X, y, C, cfg


def split_indices(n, val_ratio, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    if val_ratio <= 0:
        return idx, []
    n_val = max(1, int(round(n * val_ratio)))
    # 至少各留一条，避免验证集为空
    return idx[n_val:], idx[:n_val]


@torch.no_grad()
def evaluate(model, loader, criterion, dev):
    model.eval()
    tot_loss, tp, fp, fn, correct, total = 0.0, 0, 0, 0, 0, 0
    for bx, by, bc in loader:
        bx, by, bc = bx.to(dev), by.to(dev), bc.to(dev)
        logits, _ = model(bx, yolo_conf=bc)
        tot_loss += float(criterion(logits, by)) * by.size(0)
        pred = logits.argmax(1)
        correct += int((pred == by).sum())
        total += int(by.numel())
        tp += int(((pred == 1) & (by == 1)).sum())
        fp += int(((pred == 1) & (by == 0)).sum())
        fn += int(((pred == 0) & (by == 1)).sum())
    acc = correct / max(total, 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"loss": tot_loss / max(total, 1), "acc": acc, "precision": prec,
            "recall": rec, "f1": f1, "n": total}


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="训练溺水 TCN 分类器")
    parser.add_argument("--data", default=str(root / "weights" / "tcn" / "train.pt"),
                        help="extract_features.py 产出的 .pt（默认 weights/tcn/train.pt）")
    parser.add_argument("--out", default=str(root / "weights" / "tcn" / "tcn_drowning.pth"),
                        help="模型输出路径（默认 weights/tcn/tcn_drowning.pth）")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-gated-fusion", action="store_true", help="不启用 YOLO/TCN 门控融合头")
    parser.add_argument("--use-graph", action="store_true",
                        help="训练图交互分支（离线单目标窗口没有多目标关系，默认关闭）")
    parser.add_argument("--class-weight", dest="class_weight", action="store_true", default=True,
                        help="按类别频率加权交叉熵（默认开启）")
    parser.add_argument("--no-class-weight", dest="class_weight", action="store_false")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        args.device if args.device != "auto" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] 请求 CUDA 但当前不可用，回退 CPU")
        device = "cpu"
    dev = torch.device(device)

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"找不到特征文件 {data_path}，请先运行 extract_features.py")
    X, y, C, cfg = load_payload(data_path)
    print(f"数据 {tuple(X.shape)}  正类(溺水) {int((y == 1).sum())} / 负类 {int((y == 0).sum())}  device={dev}")
    if cfg["window_size"] != X.shape[2]:
        print(f"[WARN] window {cfg['window_size']} 与 X 时间维 {X.shape[2]} 不符，以 X 为准")
        cfg["window_size"] = int(X.shape[2])

    tr_idx, va_idx = split_indices(X.shape[0], args.val_ratio, args.seed)
    if not tr_idx:
        raise SystemExit("训练集为空，请减小 --val-ratio 或补充样本")

    def loader_for(idx, shuffle):
        ds = TensorDataset(X[idx], y[idx], C[idx])
        return DataLoader(ds, batch_size=min(args.batch_size, max(len(idx), 1)), shuffle=shuffle)

    train_loader = loader_for(tr_idx, True)
    val_loader = loader_for(va_idx, False)

    # 与推理端一致的结构（主干冻结，只训练时序与判决头）
    model = DrowningTCNModel(
        cfg["tcn_channels"], tcn_kernel_size=2, dropout=args.dropout,
        feature_dim=cfg["feature_dim"], num_classes=2, freeze_feature_extractor=True,
        use_attention=True, use_gated_fusion=not args.no_gated_fusion, use_graph=args.use_graph,
    ).to(dev)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数 {sum(p.numel() for p in trainable):,}（主干已冻结）")
    if not trainable:
        raise SystemExit("没有可训练参数")

    weights = None
    if args.class_weight:
        counts = torch.tensor([(y[tr_idx] == i).sum().item() for i in range(2)], dtype=torch.float32)
        counts[counts == 0] = 1.0
        weights = (counts.sum() / (2 * counts)).to(dev)
        print(f"类别权重 {weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(args.epochs, 1))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_metric, best_state = float("-inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for bx, by, bc in train_loader:
            bx, by, bc = bx.to(dev), by.to(dev), bc.to(dev)
            logits, _ = model(bx, yolo_conf=bc)
            loss = criterion(logits, by)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0)
            optim.step()
            run_loss += float(loss) * by.size(0)
            seen += int(by.numel())
        scheduler.step()

        train_loss = run_loss / max(seen, 1)
        if val_loader:
            m = evaluate(model, val_loader, nn.CrossEntropyLoss(), dev)
            print(f"epoch {epoch:>3}/{args.epochs}  train_loss {train_loss:.4f}  "
                  f"val_loss {m['loss']:.4f}  val_acc {m['acc']:.3f}  val_f1 {m['f1']:.3f}")
            monitor = m["f1"] if m["n"] else -m["loss"]
        else:
            print(f"epoch {epoch:>3}/{args.epochs}  train_loss {train_loss:.4f}")
            monitor = -train_loss

        if monitor > best_metric:
            best_metric, best_state = monitor, {k: v.detach().cpu().clone()
                                                for k, v in model.state_dict().items()}

    state = best_state if best_state is not None else model.state_dict()
    checkpoint = {
        "state_dict": state,
        "tcn_channels": cfg["tcn_channels"],
        "feature_dim": cfg["feature_dim"],
        "window_size": cfg["window_size"],
        "use_attention": True,
        "use_gated_fusion": not args.no_gated_fusion,
        "use_graph": args.use_graph,
        "dropout": args.dropout,
        "num_classes": 2,
        "trained_on": str(data_path),
        "metrics": json.dumps({"best_monitor": float(best_metric)}),
    }
    torch.save(checkpoint, out_path)
    print(f"\n已保存最优模型 → {out_path}")
    print("推理示例：python src/training/tcn/model_def.py --yolo-model <yolo.pt> "
          f"--tcn-model {out_path} --source 0")


if __name__ == "__main__":
    main()
