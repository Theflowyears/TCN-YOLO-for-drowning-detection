import cv2
import json
import os
import argparse
from collections import defaultdict

import numpy as np
from ultralytics import YOLO

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


class SimpleTracker:
    def __init__(self, max_age=30, min_hits=3):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.min_hits = min_hits

    def update(self, detections):
        track_ids = list(self.tracks.keys())
        pred_boxes = []
        for tid in track_ids:
            data = self.tracks[tid]
            pred = data['kf'].predict()
            pred_boxes.append([pred[0], pred[1], pred[0]+data['bbox'][2]-data['bbox'][0],
                               pred[1]+data['bbox'][3]-data['bbox'][1]])

        det_boxes = np.array([det[:4] for det in detections])
        if len(pred_boxes) > 0 and len(det_boxes) > 0:
            iou_mat = np.zeros((len(pred_boxes), len(det_boxes)))
            for i, pb in enumerate(pred_boxes):
                for j, db in enumerate(det_boxes):
                    iou_mat[i, j] = self._iou(pb, db)
        else:
            iou_mat = np.zeros((len(pred_boxes), len(det_boxes)))

        matched_tracks = set()
        matched_dets = set()
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

        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]['age'] += 1
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['age'] > self.max_age:
                del self.tracks[tid]
        for did, det in enumerate(detections):
            if did not in matched_dets:
                bbox = det[:4]
                cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                kf = KalmanFilter()
                kf.init(cx, cy)
                self.tracks[self.next_id] = {
                    'bbox': bbox,
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


def generate_tracks(video_path, yolo_model_path, output_json, conf=0.5,
                    max_age=30, min_hits=3, save_frames_dir=None, min_frames=16,
                    person_cls=0):
    """
    对视频进行检测和跟踪，生成每个人的轨迹信息。
    轨迹保存为 JSON 格式，每个 track 包含：
        - track_id: 分配的ID
        - video_name: 视频文件名
        - frames: 帧序号列表
        - bboxes: 对应的 [x1,y1,x2,y2] 列表
        - confs: 对应的检测置信度列表
    长度不足 min_frames 的轨迹会先用线性插值补帧，仍不足则丢弃。
    """
    # 初始化 YOLO
    yolo = YOLO(yolo_model_path)
    tracker = SimpleTracker(max_age=max_age, min_hits=min_hits)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频 {video_path}")

    frame_idx = 0
    # 存储轨迹的字典： track_id -> {'frames':[], 'bboxes':[], 'confs':[]}
    trajectory_data = defaultdict(lambda: {'frames': [], 'bboxes': [], 'confs': []})

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # YOLO 检测
        results = yolo(frame, conf=conf)[0]
        person_dets = []
        if results.boxes is not None:
            for box in results.boxes:
                if int(box.cls[0]) == person_cls:  # COCO 中 person 类别 ID 为 0
                    bbox = box.xyxy[0].tolist()
                    conf_val = float(box.conf[0])
                    person_dets.append(bbox + [conf_val])

        # 更新跟踪器
        tracker.update(person_dets)
        active_tracks = tracker.get_active_tracks()

        # 保存每个 track 当前帧的信息
        for tid, data in active_tracks.items():
            trajectory_data[tid]['frames'].append(frame_idx)
            trajectory_data[tid]['bboxes'].append(data['bbox'].tolist() if isinstance(data['bbox'], np.ndarray) else data['bbox'])
            trajectory_data[tid]['confs'].append(data['conf'])

        # 可选：保存带跟踪 ID 的帧图像用于检查
        if save_frames_dir:
            os.makedirs(save_frames_dir, exist_ok=True)
            draw = frame.copy()
            for tid, data in active_tracks.items():
                x1,y1,x2,y2 = map(int, data['bbox'])
                cv2.rectangle(draw, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(draw, f'ID:{tid}', (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.imwrite(os.path.join(save_frames_dir, f'frame_{frame_idx:06d}.jpg'), draw)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"处理帧 {frame_idx}")

    cap.release()

    def interpolate_bboxes(track_data):
        """用线性插值填充缺失帧的 bbox"""
        frames = track_data['frames']
        bboxes = track_data['bboxes']
        confs = track_data['confs']
        if len(frames) < 2:
            return
        new_frames = [frames[0]]
        new_bboxes = [bboxes[0]]
        new_confs = [confs[0]]
        for i in range(1, len(frames)):
            prev_f = frames[i - 1]
            curr_f = frames[i]
            if curr_f - prev_f > 1:
                # 中间有缺失帧，线性插值
                for missing_f in range(prev_f + 1, curr_f):
                    alpha = (missing_f - prev_f) / (curr_f - prev_f)
                    x1 = bboxes[i - 1][0] + (bboxes[i][0] - bboxes[i - 1][0]) * alpha
                    y1 = bboxes[i - 1][1] + (bboxes[i][1] - bboxes[i - 1][1]) * alpha
                    x2 = bboxes[i - 1][2] + (bboxes[i][2] - bboxes[i - 1][2]) * alpha
                    y2 = bboxes[i - 1][3] + (bboxes[i][3] - bboxes[i - 1][3]) * alpha
                    new_frames.append(missing_f)
                    new_bboxes.append([x1, y1, x2, y2])
                    new_confs.append(confs[i - 1])  # 置信度沿用前一帧
            new_frames.append(curr_f)
            new_bboxes.append(bboxes[i])
            new_confs.append(confs[i])
        track_data['frames'] = new_frames
        track_data['bboxes'] = new_bboxes
        track_data['confs'] = new_confs

    # 短轨迹先补插值，再按最小帧数过滤，最后统一整理输出结构
    output_data = []
    for tid, data in trajectory_data.items():
        if len(data['frames']) < min_frames:
            interpolate_bboxes(data)
        if len(data['frames']) < min_frames:
            continue
        output_data.append({
            "track_id": tid,
            "video_name": os.path.basename(video_path),
            "frames": data['frames'],
            "bboxes": data['bboxes'],
            "confs": data['confs']
        })

    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"轨迹已保存至 {output_json}，共 {len(output_data)} 条有效轨迹（>={min_frames}帧）")

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='输入视频路径')
    parser.add_argument('--yolo-model', required=True, help='YOLO 模型路径，如 yolov8n.pt')
    parser.add_argument('--output', required=True, help='输出 JSON 文件路径')
    parser.add_argument('--conf', type=float, default=0.5, help='YOLO 置信度阈值')
    parser.add_argument('--max-age', type=int, default=30, help='跟踪最大丢失帧数')
    parser.add_argument('--min-hits', type=int, default=3, help='轨迹有效最小出现帧数')
    parser.add_argument('--save-frames', default=None, help='可选，保存跟踪可视化帧的目录')
    parser.add_argument('--min-frames', type=int, default=16,
                        help='轨迹保留的最小帧数，应不低于 TCN 特征窗口长度（默认 16）')
    parser.add_argument('--person-cls', type=int, default=0,
                        help='YOLO 模型中作为跟踪目标的人员类别 ID（COCO 为 0）')
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    generate_tracks(args.video, args.yolo_model, args.output,
                    conf=args.conf, max_age=args.max_age, min_hits=args.min_hits,
                    save_frames_dir=args.save_frames, min_frames=args.min_frames,
                    person_cls=args.person_cls)