import cv2
import argparse
import logging
import os
import json
from datetime import datetime
from collections import deque, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from ultralytics import YOLO

# 尝试导入绘图库（非必需）
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def ensure_jit_scriptable(module):
    """让任意 nn.Module 子树在 Python 3.14 上重新可被 torch.jit.script。

    Python 3.14（PEP 649）不再为实例暴露 ``__annotations__``，而 torch 2.9 的
    TorchScript ``AttributeTypeIsSupportedChecker`` 会直接读取它，于是
    ``torch.jit.script`` 对**任何**模块都会抛
    ``AttributeError: 'Xxx' object has no attribute '__annotations__'``
    （连 ``torch.nn.Linear`` 单独脚本化都会失败）。

    这里在脚本化之前给子树里每个缺失该属性的实例补一个空注解字典，绕开检查器。
    属于纯兼容性 workaround：不改变任何算子与参数。

    :return: 实际修补的模块实例数量
    """
    patched = 0
    for mod in module.modules():
        try:
            mod.__annotations__
        except AttributeError:
            object.__setattr__(mod, "__annotations__", {})   # 绕过 nn.Module.__setattr__
            patched += 1
    return patched


# ------------------------------------------------------------
# 1. 注意力与基础模块（通道+空间 CBAM-1D）
# ------------------------------------------------------------
class ChannelAttention1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * self.sigmoid(out).view(b, c, 1)


class SpatialAttention1D(nn.Module):
    """对时间维度（L）做注意力"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class CBAM1D(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention1D(channel, reduction)
        self.spatial_att = SpatialAttention1D(spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ------------------------------------------------------------
# 2. 因果卷积与多尺度 TCN 块（代替 Chomp1d，使用 padding + 切片）
# ------------------------------------------------------------
class CausalConv1d(nn.Module):
    """因果卷积，保证只依赖过去信息"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))          # 只在前端填充
        return self.conv(x)


class MultiScaleTemporalBlock(nn.Module):
    """
    多尺度膨胀卷积 + 深度可分离 + CBAM 注意力 + 残差连接
    """
    def __init__(self, n_inputs, n_outputs, kernel_size=2, dropout=0.2,
                 dilations=(1, 2, 4), use_attention=True):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            branch = nn.Sequential(
                # 深度卷积必须保持 out=in（groups 才能等于 n_inputs），
                # 随后用 1x1 逐点卷积把通道映射到 n_outputs
                CausalConv1d(n_inputs, n_inputs, kernel_size, dilation=d, groups=n_inputs),
                nn.Conv1d(n_inputs, n_outputs, 1),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                CausalConv1d(n_outputs, n_outputs, kernel_size, dilation=d, groups=n_outputs),
                nn.Conv1d(n_outputs, n_outputs, 1),
            )
            self.branches.append(branch)

        self.fuse = nn.Conv1d(len(dilations) * n_outputs, n_outputs, 1)
        self.att = CBAM1D(n_outputs) if use_attention else nn.Identity()
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outs = []
        for branch in self.branches:
            outs.append(branch(x))
        out = torch.cat(outs, dim=1)
        out = self.fuse(out)
        out = self.att(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# ------------------------------------------------------------
# 3. 动态门控融合模块（替代原全连接融合）
# ------------------------------------------------------------
class GatedFusionModule(nn.Module):
    """门控机制融合 YOLO 置信度与 TCN 概率"""
    def __init__(self, yolo_dim=1, tcn_dim=2, hidden=16, num_classes=2):
        super().__init__()
        self.linear_yolo = nn.Linear(yolo_dim, hidden)
        self.linear_tcn = nn.Linear(tcn_dim, hidden)
        self.gate = nn.Sequential(
            nn.Linear(yolo_dim + tcn_dim, hidden),
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, yolo_conf, tcn_prob):
        # yolo_conf: (B,1), tcn_prob: (B,2)
        feat_yolo = self.linear_yolo(yolo_conf)
        feat_tcn = self.linear_tcn(tcn_prob)
        gate = self.gate(torch.cat([yolo_conf, tcn_prob], dim=1))
        fused = gate * feat_yolo + (1 - gate) * feat_tcn
        return self.classifier(fused)


# ------------------------------------------------------------
# 4. 图交互模块（建模目标间关系）
# ------------------------------------------------------------
class GraphInteraction(nn.Module):
    def __init__(self, feat_dim=128, hidden_dim=64):
        super().__init__()
        # 输入为 [节点特征 | 加权邻居消息] 拼接，维度为 2*feat_dim
        self.node_update = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )

    def forward(self, node_feats, boxes):
        """
        node_feats: (N, feat_dim) 每个目标的特征
        boxes: (N, 4) bbox [x1,y1,x2,y2]
        """
        N = node_feats.size(0)
        if N <= 1:
            return node_feats
        # 构建距离特征
        centers = torch.stack([(boxes[:,0]+boxes[:,2])/2, (boxes[:,1]+boxes[:,3])/2], dim=1)  # N,2
        dist = torch.cdist(centers, centers)  # N,N
        adj = torch.exp(-dist / 100.0)         # 简单的高斯核
        # 消息传递
        messages = []
        for i in range(N):
            neigh_weights = adj[i].unsqueeze(1)  # N,1
            neighbor_sum = (node_feats * neigh_weights).sum(0, keepdim=True)  # 1,F
            messages.append(neighbor_sum)
        messages = torch.cat(messages, dim=0)  # N,F
        updated = self.node_update(torch.cat([node_feats, messages], dim=1))
        return updated


# ------------------------------------------------------------
# 4.5 可脚本化的序列分类子模块（供 TorchScript 推理加速）
# ------------------------------------------------------------
class TCNSequenceClassifier(nn.Module):
    """封装 TCN + 图交互 + 分类头 + 门控融合的批量序列分类器。

    与 DrowningTCNModel 共享子模块实例（参数为同一份），
    可被 torch.jit.script 完整脚本化，用于推理加速
    （Python 3.14 下需先调用 ensure_jit_scriptable()，见该函数文档）。
    """

    def __init__(self, tcn, tcn_classifier, graph, fusion):
        super().__init__()
        self.tcn = tcn
        self.tcn_classifier = tcn_classifier
        self.graph = graph
        self.fusion = fusion

    def forward(self, seqs, yolo_confs, boxes):
        # seqs: (N, feature_dim, T)，yolo_confs: (N, 1)，boxes: (N, 4)
        tcn_out = self.tcn(seqs)
        last_feat = tcn_out[:, :, -1]  # (N, C)
        if self.graph is not None and last_feat.size(0) > 1:
            last_feat = self.graph(last_feat, boxes)
        tcn_logits = self.tcn_classifier(last_feat)
        tcn_probs = F.softmax(tcn_logits, dim=1)
        if self.fusion is not None:
            fusion_logits = self.fusion(yolo_confs, tcn_probs)
            return fusion_logits, F.softmax(fusion_logits, dim=1), tcn_probs
        return tcn_logits, tcn_probs, tcn_probs


# ------------------------------------------------------------
# 5. 带硬件优化的溺水识别模型（MobileNetV2 + TCN）
# ------------------------------------------------------------
class DrowningTCNModel(nn.Module):
    def __init__(self, tcn_channels, tcn_kernel_size=2, dropout=0.2,
                 feature_dim=1280, num_classes=2, freeze_feature_extractor=True,
                 use_attention=True, use_gated_fusion=True, use_graph=True):
        super().__init__()
        backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.feature_extractor = backbone.features
        if freeze_feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 多尺度 TCN
        tcn_blocks = []
        num_levels = len(tcn_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = feature_dim if i == 0 else tcn_channels[i - 1]
            out_ch = tcn_channels[i]
            tcn_blocks.append(MultiScaleTemporalBlock(
                in_ch, out_ch, kernel_size=tcn_kernel_size, dropout=dropout,
                dilations=(1, 2, 4), use_attention=use_attention))
        self.tcn = nn.Sequential(*tcn_blocks)

        self.tcn_classifier = nn.Linear(tcn_channels[-1], num_classes)
        self.fusion = GatedFusionModule() if use_gated_fusion else None
        self.graph = GraphInteraction(feat_dim=tcn_channels[-1]) if use_graph else None

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        # 序列分类子模块（与上面的子模块共享参数，可整体转 TorchScript）
        self.sequence_classifier = TCNSequenceClassifier(
            self.tcn, self.tcn_classifier, self.graph, self.fusion)
        self.scripted_classifier = None

        # 预处理参数（自适应裁剪在外部实现，此处保留 transform 作为后备）
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_features_from_bbox(self, frame, bbox, flow_frame=None):
        """自适应裁剪 + 可选光流拼接"""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        # 扩大 10% 并保持宽高比，letterbox 处理
        cx, cy = (x1+x2)/2, (y1+y2)/2
        bw, bh = x2-x1, y2-y1
        scale = 1.2
        nw, nh = bw*scale, bh*scale
        # 正方形最长边
        side = max(nw, nh)
        nw = nh = side
        # 新的边界
        nx1 = max(0, int(cx - nw/2))
        ny1 = max(0, int(cy - nh/2))
        nx2 = min(w, int(cx + nw/2))
        ny2 = min(h, int(cy + nh/2))
        if nx2 <= nx1 or ny2 <= ny1:
            return torch.zeros(self.feature_dim,
                               device=next(self.feature_extractor.parameters()).device)
        roi = frame[ny1:ny2, nx1:nx2]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # letterbox resize
        roi_resized = self._letterbox_resize(roi, (224, 224))
        device = next(self.feature_extractor.parameters()).device
        x = self._preprocess(roi_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = self.feature_extractor(x)
            feat = self.pool(feat)
            feat = feat.view(1, -1)
        feat = feat.squeeze(0)

        # 光流特征（可选）。光流是 (H,W,2) 的 float 图，无法与 3 通道 RGB 直接 concat
        # （那样会让 MobileNetV2 报通道数错误），因此单独编码成 3 通道、过同一主干，
        # 再与 RGB 特征取均值，保持 feature_dim 不变。
        if flow_frame is not None:
            flow_roi = self._encode_flow(flow_frame[ny1:ny2, nx1:nx2])
            if flow_roi is not None:
                flow_roi = cv2.resize(flow_roi, (224, 224), interpolation=cv2.INTER_LINEAR)
                fx = self._preprocess(flow_roi).unsqueeze(0).to(device)
                with torch.no_grad():
                    ffeat = self.pool(self.feature_extractor(fx)).view(1, -1).squeeze(0)
                feat = 0.5 * (feat + ffeat)
        return feat

    def _preprocess(self, rgb_uint8):
        """uint8 RGB 图 → ImageNet 归一化的 float tensor（3, 224, 224）。"""
        t = transforms.ToTensor()(rgb_uint8)
        return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])(t)

    @staticmethod
    def _encode_flow(flow_roi):
        """(H,W,2) magnitude/angle 光流 → 3 通道 uint8，供 RGB 主干消费。"""
        if flow_roi is None or flow_roi.size == 0:
            return None
        if flow_roi.ndim == 2:
            flow_roi = np.stack([flow_roi, np.zeros_like(flow_roi)], axis=2)
        mag = flow_roi[:, :, 0].astype(np.float32)
        ang = flow_roi[:, :, 1].astype(np.float32)
        mag_u8 = np.clip(mag / (float(mag.max()) + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        ang_u8 = np.clip((ang + np.pi) / (2 * np.pi) * 255.0, 0, 255).astype(np.uint8)
        return np.stack([mag_u8, ang_u8, np.zeros_like(mag_u8)], axis=2)

    def _letterbox_resize(self, img, target_size):
        h, w = img.shape[:2]
        th, tw = target_size
        scale = min(th/h, tw/w)
        nh, nw = int(h*scale), int(w*scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        dx, dy = (tw - nw)//2, (th - nh)//2
        canvas[dy:dy+nh, dx:dx+nw] = resized
        return canvas

    def forward(self, x, yolo_conf=None, boxes=None):
        # x: (B, feature_dim, T)
        fusion_logits, fusion_probs, _ = self.classify_sequences(x, yolo_conf, boxes)
        return fusion_logits, fusion_probs

    def classify_sequences(self, seqs, yolo_confs=None, boxes=None):
        """批量序列分类：TCN -> 可选图交互 -> 分类头 -> 可选门控融合。

        返回 (fusion_logits, fusion_probs, tcn_probs)；
        未启用门控融合时，前两项即 TCN 分类结果。
        当 yolo_confs/boxes 齐全且已加载脚本化分类器时走 TorchScript 路径。
        """
        if yolo_confs is not None and boxes is not None:
            if self.scripted_classifier is not None:
                return self.scripted_classifier(seqs, yolo_confs, boxes)
            return self.sequence_classifier(seqs, yolo_confs, boxes)
        # 输入不完整（缺图/缺融合输入）时的纯 Python 兜底路径
        tcn_out = self.tcn(seqs)
        last_feat = tcn_out[:, :, -1]
        if self.graph is not None and boxes is not None and last_feat.size(0) > 1:
            last_feat = self.graph(last_feat, boxes)
        tcn_logits = self.tcn_classifier(last_feat)
        tcn_probs = F.softmax(tcn_logits, dim=1)
        if self.fusion is not None and yolo_confs is not None:
            fusion_logits = self.fusion(yolo_confs, tcn_probs)
            return fusion_logits, F.softmax(fusion_logits, dim=1), tcn_probs
        return tcn_logits, tcn_probs, tcn_probs

    def set_scripted_classifier(self, scripted):
        """注入脚本化（TorchScript）序列分类器，用于推理加速。"""
        self.scripted_classifier = scripted


# ------------------------------------------------------------
# 6. 卡尔曼滤波器
# ------------------------------------------------------------
class KalmanFilter:
    """简单恒速模型，状态[x, y, dx, dy]"""
    def __init__(self):
        self.dt = 1.0
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 1.0
        self.x = None
        self.P = np.eye(4) * 10.0

    def init(self, x, y):
        self.x = np.array([x, y, 0, 0], dtype=np.float32)

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.H @ self.x

    def update(self, z):
        z = np.array(z, dtype=np.float32)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P


# ------------------------------------------------------------
# 7. 跟踪器（卡尔曼 + IoU 匹配）
# ------------------------------------------------------------
class SimpleTracker:
    def __init__(self, max_age=30, min_hits=3):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.min_hits = min_hits

    def update(self, detections):
        """detections: list of [x1, y1, x2, y2, conf]"""
        # 预测所有 track 位置
        track_ids = list(self.tracks.keys())
        pred_boxes = []
        for tid in track_ids:
            data = self.tracks[tid]
            pred = data['kf'].predict()
            pred_boxes.append([pred[0], pred[1], pred[0]+data['bbox'][2]-data['bbox'][0],
                               pred[1]+data['bbox'][3]-data['bbox'][1]])

        det_boxes = np.array([det[:4] for det in detections])
        # IoU 矩阵
        if len(pred_boxes) > 0 and len(det_boxes) > 0:
            iou_mat = np.zeros((len(pred_boxes), len(det_boxes)))
            for i, pb in enumerate(pred_boxes):
                for j, db in enumerate(det_boxes):
                    iou_mat[i, j] = self._iou(pb, db)
        else:
            iou_mat = np.zeros((len(pred_boxes), len(det_boxes)))

        matched_tracks = set()
        matched_dets = set()
        # 匹配
        for _ in range(min(len(pred_boxes), len(det_boxes))):
            if iou_mat.size == 0:
                break
            idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            if iou_mat[idx] < 0.3:
                break
            tid = track_ids[idx[0]]
            did = idx[1]
            z = [(det_boxes[did, 0]+det_boxes[did, 2])/2,
                 (det_boxes[did, 1]+det_boxes[did, 3])/2]
            self.tracks[tid]['kf'].update(z)
            self.tracks[tid]['bbox'] = det_boxes[did]
            self.tracks[tid]['hits'] += 1
            self.tracks[tid]['age'] = 0
            self.tracks[tid]['conf'] = detections[did][4]
            matched_tracks.add(tid)
            matched_dets.add(did)
            iou_mat[idx[0], :] = -1
            iou_mat[:, idx[1]] = -1

        # 未匹配 track age+1
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]['age'] += 1
        # 删除超龄
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['age'] > self.max_age:
                del self.tracks[tid]
        # 新检测新建 track
        for did, det in enumerate(detections):
            if did not in matched_dets:
                bbox = det[:4]
                cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                kf = KalmanFilter()
                kf.init(cx, cy)
                self.tracks[self.next_id] = {
                    'bbox': bbox,
                    'buffer': deque(maxlen=16),
                    'hits': 1,
                    'age': 0,
                    'conf': det[4],
                    'kf': kf
                }
                self.next_id += 1

    def _iou(self, a, b):
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        areaA = (a[2]-a[0])*(a[3]-a[1])
        areaB = (b[2]-b[0])*(b[3]-b[1])
        return inter / (areaA + areaB - inter + 1e-6)

    def get_active_tracks(self):
        return {tid: data for tid, data in self.tracks.items() if data['hits'] >= self.min_hits}


# ------------------------------------------------------------
# 8. 主检测器：YOLO + TCN 多模态系统
# ------------------------------------------------------------
class DrowningDetector:
    def __init__(self, yolo_model_path, tcn_model_path, conf_threshold=0.5,
                 window_size=16, alert_callback=None, save_dir=None,
                 output_dir=None, device='cuda', use_quantization=False,
                 use_torchscript=False, use_flow=False, use_graph=True):
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.logger = self._setup_logger()
        self.yolo = YOLO(yolo_model_path)
        self.logger.info(f"YOLO 模型已加载: {yolo_model_path}")

        # 先读 checkpoint：结构超参以训练时写入的为准，
        # 否则会出现「训练一套结构、推理另一套结构」而 strict=False 把失配静默掉。
        ckpt, state = self._read_tcn_checkpoint(tcn_model_path, self.device)
        tcn_channels = list(ckpt.get('tcn_channels', [64, 64, 64, 64]))
        self.tcn_model = DrowningTCNModel(tcn_channels, tcn_kernel_size=2,
                                          dropout=float(ckpt.get('dropout', 0.2)),
                                          feature_dim=int(ckpt.get('feature_dim', 1280)),
                                          num_classes=int(ckpt.get('num_classes', 2)),
                                          freeze_feature_extractor=True,
                                          use_attention=bool(ckpt.get('use_attention', True)),
                                          use_gated_fusion=bool(ckpt.get('use_gated_fusion', True)),
                                          use_graph=bool(ckpt.get('use_graph', use_graph)))
        self._apply_tcn_state(tcn_model_path, ckpt, state)
        self.tcn_model.to(self.device)
        self.tcn_model.eval()
        ckpt_window = ckpt.get('window_size')

        # 量化
        if use_quantization and self.device.type == 'cpu':
            self.tcn_model = torch.quantization.quantize_dynamic(
                self.tcn_model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
            )
            self.logger.info("已应用动态量化")

        # TorchScript 优化（只脚本化序列分类子模块，主模型保留 Python 接口）
        self.use_torchscript = use_torchscript
        if use_torchscript:
            try:
                fixed = ensure_jit_scriptable(self.tcn_model.sequence_classifier)
                scripted = torch.jit.script(self.tcn_model.sequence_classifier)
                self.tcn_model.set_scripted_classifier(scripted)
                self.logger.info(f"序列分类器已转为 TorchScript（兼容性修补 {fixed} 个模块）")
            except Exception as e:
                self.logger.warning(f"TorchScript 转换失败: {e}，回退到普通模型")

        self.conf_threshold = conf_threshold
        if ckpt_window and int(ckpt_window) != window_size:
            self.logger.warning(
                f"特征窗口以 checkpoint 为准：{window_size} → {int(ckpt_window)}")
            window_size = int(ckpt_window)
        self.window_size = window_size
        self.alert_callback = alert_callback
        self.tracker = SimpleTracker(max_age=30, min_hits=3)
        self.feature_buffers = OrderedDict()
        self.use_flow = use_flow
        self.prev_gray = None

        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        # 统计
        self.detection_stats = {
            'total_frames': 0,
            'drowning_events': 0,
            'drowning_timestamps': [],
            'confidences_over_time': [],
            'start_time': datetime.now(),
            'end_time': None
        }
        # 在线学习缓存
        self.online_buffer = []

    @staticmethod
    def _read_tcn_checkpoint(path, device):
        """安全读取 TCN checkpoint，返回 (ckpt, state_dict)。

        优先使用 ``weights_only=True``（可执行任意代码的 pickle 会被拒），
        只有旧版含非张量字段的 checkpoint 才回退。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"TCN 权重不存在: {path}")
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(f"TCN 权重格式不正确（期望 dict）: {path}")
        return ckpt, ckpt.get('state_dict', ckpt)

    def _apply_tcn_state(self, path, ckpt, state):
        """加载权重并显式报告缺失/多余键，避免静默使用随机初始化。"""
        missing, unexpected = self.tcn_model.load_state_dict(state, strict=False)
        # 主干（feature_extractor.*）在只训练 TCN 时也可能缺失，属于预期内
        head_missing = [k for k in missing if not k.startswith('feature_extractor.')]
        if head_missing:
            self.logger.warning(f"TCN 判决/时序分支有 {len(head_missing)} 个参数未从 checkpoint 加载"
                                f"（仍为随机初始化），例如: {head_missing[:5]}")
        if unexpected:
            self.logger.warning(f"checkpoint 中有 {len(unexpected)} 个未使用的键，例如: {unexpected[:5]}")
        self.logger.info(f"TCN 权重已加载: {path}")

    def _setup_logger(self):
        logger = logging.getLogger('DrowningDetector')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(ch)
        return logger

    def _compute_flow(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros((*gray.shape, 2), dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev_gray = gray
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        return np.stack([mag, ang], axis=2)  # H,W,2

    def detect(self, frame):
        """返回 detections 列表和是否溺水"""
        self.detection_stats['total_frames'] += 1
        # 光流
        flow_frame = self._compute_flow(frame) if self.use_flow else None

        yolo_res = self.yolo(frame, conf=self.conf_threshold)[0]
        person_dets = []
        if yolo_res.boxes is not None:
            for box in yolo_res.boxes:
                if int(box.cls[0]) == 0:  # person
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    person_dets.append(xyxy + [conf])
        self.tracker.update(person_dets)
        active = self.tracker.get_active_tracks()

        detections = []
        drowning_detected = False

        # 先提取所有 track 的特征
        tid_feat_map = {}
        for tid, data in active.items():
            bbox = data['bbox']
            feat = self.tcn_model.extract_features_from_bbox(frame, bbox, flow_frame)
            if feat.sum() == 0:
                continue
            if tid not in self.feature_buffers:
                self.feature_buffers[tid] = deque(maxlen=self.window_size)
            self.feature_buffers[tid].append(feat)

            if len(self.feature_buffers[tid]) == self.window_size:
                seq = torch.stack(list(self.feature_buffers[tid]), dim=1).unsqueeze(0).to(self.device)
                tid_feat_map[tid] = seq

        # 批量序列分类：TCN -> (图交互) -> 分类头 -> 门控融合
        if tid_feat_map:
            tids = list(tid_feat_map.keys())
            seqs = torch.cat([tid_feat_map[t] for t in tids], dim=0)  # (N, feature_dim, T)
            yolo_confs = torch.tensor([[active[t]['conf']] for t in tids], device=self.device)
            box_tensor = torch.tensor([active[t]['bbox'] for t in tids], device=self.device)
            with torch.no_grad():
                _, fusion_probs, _ = self.tcn_model.classify_sequences(
                    seqs, yolo_confs=yolo_confs, boxes=box_tensor)

            for i, tid in enumerate(tids):
                prob = fusion_probs[i]
                pred_idx = torch.argmax(prob).item()
                conf = prob[pred_idx].item()
                class_name = "Drowning" if pred_idx == 1 else "Normal"
                if class_name == "Drowning" and conf >= self.conf_threshold:
                    drowning_detected = True
                    self.detection_stats['drowning_events'] += 1
                    self.detection_stats['drowning_timestamps'].append(datetime.now().isoformat())
                detections.append({
                    'bbox': active[tid]['bbox'],
                    'confidence': conf,
                    'class_id': pred_idx,
                    'class_name': class_name,
                    'track_id': tid
                })
                # 记录最高溺水置信度用于绘图
                self.detection_stats['confidences_over_time'].append(
                    (self.detection_stats['total_frames'], prob[1].item())
                )

        # 清理缓冲区
        self.feature_buffers = OrderedDict(
            (tid, buf) for tid, buf in self.feature_buffers.items() if tid in active
        )
        return detections, drowning_detected

    def process_frame(self, frame, frame_idx=None):
        detections, drowning = self.detect(frame)
        if drowning and self.alert_callback:
            self.save_drowning_frame(frame, detections, frame_idx)
            self.alert_callback(frame, detections, None)

        display = frame.copy()
        for det in detections:
            x1,y1,x2,y2 = map(int, det['bbox'])
            color = (0,0,255) if det['class_name']=='Drowning' else (0,255,0)
            label = f"{det['class_name']}:{det['confidence']:.2f}"
            cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
            cv2.putText(display, label, (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        h,w = display.shape[:2]
        cv2.putText(display, f"Events:{self.detection_stats['drowning_events']}",
                    (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return display, len(detections)

    def save_drowning_frame(self, frame, detections, idx):
        if not self.save_dir:
            return
        annotated = frame.copy()
        for d in detections:
            if d['class_name'] != 'Drowning': continue
            x1,y1,x2,y2 = map(int, d['bbox'])
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255),2)
        fname = f"drowning_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_f{idx}.jpg"
        cv2.imwrite(os.path.join(self.save_dir, fname), annotated)

    def run_on_video(self, source, output_path=None, display=True):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.logger.error("无法打开视频源")
            return
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h)) if output_path else None
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                processed, _ = self.process_frame(frame, frame_idx)
                if out: out.write(processed)
                if display:
                    cv2.imshow('Drowning Detection', processed)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                frame_idx += 1
        finally:
            cap.release()
            if out: out.release()
            cv2.destroyAllWindows()
            self.generate_report()
            self.logger.info("处理完成")

    def generate_report(self):
        self.detection_stats['end_time'] = datetime.now()
        dur = (self.detection_stats['end_time'] - self.detection_stats['start_time']).total_seconds()
        report = {
            'duration_s': dur,
            'total_frames': self.detection_stats['total_frames'],
            'drowning_events': self.detection_stats['drowning_events'],
            'drowning_timestamps': self.detection_stats['drowning_timestamps'],
            'start': self.detection_stats['start_time'].isoformat(),
            'end': self.detection_stats['end_time'].isoformat()
        }
        if self.output_dir:
            json_path = os.path.join(self.output_dir, f"report_{datetime.now():%Y%m%d_%H%M%S}.json")
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"报告已保存: {json_path}")

            # 可视化
            if HAS_MATPLOTLIB and self.detection_stats['confidences_over_time']:
                self._plot_confidence_curve()

    def _plot_confidence_curve(self):
        frames, confs = zip(*self.detection_stats['confidences_over_time'])
        plt.figure()
        plt.plot(frames, confs, label='Drowning confidence')
        plt.xlabel('Frame')
        plt.ylabel('Confidence')
        plt.title('Drowning Detection Confidence Over Time')
        plt.legend()
        path = os.path.join(self.output_dir, f"confidence_curve_{datetime.now():%Y%m%d_%H%M%S}.png")
        plt.savefig(path)
        plt.close()
        self.logger.info(f"置信度曲线已保存: {path}")

    # ---------- 在线学习接口 ----------
    def update_online(self, frames, targets, epochs=1, lr=1e-4):
        """在线微调 TCN 与分类/融合头（特征提取器保持冻结）。

        targets: 与 frames 一一对应的列表，每项为 dict：
            {'bbox': [x1, y1, x2, y2], 'class_id': 0 或 1, 'track_id': 可选}
        每个样本用 bbox 提取特征并拼成 window_size 序列，交叉熵损失 + Adam 更新。
        """
        if not frames or len(frames) != len(targets):
            self.logger.warning("update_online 要求 frames 与 targets 长度一致且非空，已跳过")
            return
        samples = [(f, t) for f, t in zip(frames, targets)
                   if isinstance(t, dict) and t.get('bbox') is not None
                   and t.get('class_id') is not None]
        if not samples:
            self.logger.warning("update_online 没有包含 bbox/class_id 的有效样本，已跳过")
            return
        trainable = [p for p in self.tcn_model.parameters() if p.requires_grad]
        if not trainable:
            self.logger.warning("update_online 无可训练参数，已跳过")
            return

        self.tcn_model.train()
        optimizer = torch.optim.Adam(trainable, lr=lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            total_loss, count = 0.0, 0
            for frame, target in samples:
                bbox = [int(v) for v in target['bbox']]
                label = torch.tensor([int(target['class_id'])], device=self.device)
                feat = self.tcn_model.extract_features_from_bbox(frame, bbox)
                if feat.sum() == 0:
                    continue
                # 单帧特征重复为 window_size 序列，使梯度可回传至 TCN 层
                seq = feat.view(1, -1, 1).repeat(1, 1, self.window_size)
                logits, _ = self.tcn_model(seq)
                loss = criterion(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1
            if count:
                self.logger.info(f"在线更新 epoch {epoch + 1}/{epochs}: loss={total_loss / count:.4f}")
        self.tcn_model.eval()

        # 脚本化分类器持有参数副本，在线更新后需重新生成
        if self.use_torchscript and self.tcn_model.scripted_classifier is not None:
            try:
                ensure_jit_scriptable(self.tcn_model.sequence_classifier)
                self.tcn_model.set_scripted_classifier(
                    torch.jit.script(self.tcn_model.sequence_classifier))
                self.logger.info("在线更新后已重新生成 TorchScript 序列分类器")
            except Exception as e:
                self.logger.warning(f"重新生成 TorchScript 失败: {e}，继续使用 Python 版本")


def default_alert_callback(frame, detections, saved_path=None):
    print(f"警报！检测到溺水！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO+TCN 溺水检测")
    parser.add_argument('--yolo-model', required=True)
    parser.add_argument('--tcn-model', required=True)
    parser.add_argument('--source', default='0')
    parser.add_argument('--output', default=None)
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--window-size', type=int, default=16)
    parser.add_argument('--save-dir', default='./drowning_detections')
    parser.add_argument('--output-dir', default='./reports')
    parser.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--torchscript', action='store_true', help='启用 TorchScript')
    parser.add_argument('--flow', action='store_true', help='启用光流多模态')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--no-graph', action='store_true', help='禁用图交互')
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    detector = DrowningDetector(
        yolo_model_path=args.yolo_model,
        tcn_model_path=args.tcn_model,
        conf_threshold=args.conf,
        window_size=args.window_size,
        alert_callback=default_alert_callback,
        save_dir=args.save_dir,
        output_dir=args.output_dir,
        device=args.device,
        use_quantization=args.quantize,
        use_torchscript=args.torchscript,
        use_flow=args.flow,
        use_graph=not args.no_graph
    )
    detector.run_on_video(source, args.output, display=not args.no_display)