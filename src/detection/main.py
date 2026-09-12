import cv2
import argparse
import logging
import os
import json
import csv
from datetime import datetime
from ultralytics import YOLO


class DrowningDetector:
    def __init__(self, model_path, conf_threshold=0.7, alert_callback=None, img_size=1280, save_dir=None,
                 output_dir=None):
        """
        :param model_path: YOLO模型文件路径 (例如 best.pt)
        :param conf_threshold: 置信度阈值
        :param alert_callback: 警报回调函数，接收 (frame, detections, saved_path) 参数
        :param save_dir: 保存溺水检测图片的目录路径
        :param output_dir: 输出结果的目录路径
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.img_size = img_size          # --imgsz 之前被解析后直接丢弃，这里真正接上
        self.alert_callback = alert_callback
        self.logger = self._setup_logger()
        self.class_names = self.model.names

        # 设置保存溺水检测图片的目录
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            self.logger.info(f"溺水检测图片将保存到: {self.save_dir}")

        # 设置输出目录
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self.logger.info(f"输出结果将保存到: {self.output_dir}")

        # 初始化统计数据
        self.detection_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'drowning_detections': 0,
            'other_detections': 0,
            'drowning_events': 0,
            'start_time': datetime.now(),
            'end_time': None,
            'detection_history': []
        }

    def _setup_logger(self):
        logger = logging.getLogger('DrowningDetector')
        logger.setLevel(logging.INFO)
        if not logger.handlers:          # 避免多次实例化时重复添加 handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def detect(self, frame):
        """对单帧进行目标检测"""
        results = self.model(frame, conf=self.conf_threshold, imgsz=self.img_size)[0]
        detections = []
        drowning_detected = False

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                class_name = self.class_names[cls_id]

                detections.append({
                    'bbox': xyxy,
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': class_name
                })

                if class_name == 'Drowning' and conf >= self.conf_threshold:
                    drowning_detected = True

        return detections, drowning_detected

    def save_drowning_frame(self, frame, detections, frame_idx=None):
        """保存溺水检测帧到指定目录（带检测框）"""
        if not self.save_dir:
            return None

        # 创建带检测框的图像
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det['bbox']

            # 确保坐标是像素值（不是归一化值）
            if max(x1, y1, x2, y2) <= 1.0:
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h

            x1 = int(max(0, min(w, x1)))
            y1 = int(max(0, min(h, y1)))
            x2 = int(max(0, min(w, x2)))
            y2 = int(max(0, min(h, y2)))

            class_name = det['class_name']
            label = f"{class_name}: {det['confidence']:.2f}"
            color = (0, 0, 255) if class_name == 'Drowning' else (0, 255, 0)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(annotated_frame, label, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 添加统计信息
        stats_text = f"Frame: {frame_idx} | Drowning: {self.detection_stats['drowning_detections']}"
        cv2.putText(annotated_frame, stats_text, (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_info = f"_frame_{frame_idx}" if frame_idx is not None else ""
        filename = f"drowning_{timestamp}{frame_info}.jpg"
        filepath = os.path.join(self.save_dir, filename)

        success = cv2.imwrite(filepath, annotated_frame)
        if success:
            self.logger.info(f"已保存溺水检测图片（带检测框）: {filepath}")
            return filepath
        else:
            self.logger.error(f"保存图片失败: {filepath}")
            return None

    def update_detection_stats(self, detections, drowning_detected, frame_idx):
        """更新检测统计数据"""
        self.detection_stats['total_frames'] += 1
        self.detection_stats['total_detections'] += len(detections)

        drowning_count = sum(1 for d in detections if d['class_name'] == 'Drowning')
        other_count = len(detections) - drowning_count

        self.detection_stats['drowning_detections'] += drowning_count
        self.detection_stats['other_detections'] += other_count

        if drowning_detected:
            self.detection_stats['drowning_events'] += 1

            # 记录检测历史
            detection_record = {
                'timestamp': datetime.now().isoformat(),
                'frame_index': frame_idx,
                'drowning_count': drowning_count,
                'total_detections': len(detections),
                'detections': [
                    {
                        'class_name': d['class_name'],
                        'confidence': float(d['confidence']),
                        'bbox': [float(coord) for coord in d['bbox']]
                    } for d in detections
                ]
            }
            self.detection_stats['detection_history'].append(detection_record)

    def generate_report(self):
        """生成检测报告"""
        if not self.output_dir:
            return None

        self.detection_stats['end_time'] = datetime.now()
        duration = (self.detection_stats['end_time'] - self.detection_stats['start_time']).total_seconds()

        # 生成JSON报告
        json_report = {
            'summary': {
                'total_processing_time_seconds': round(duration, 2),
                'total_frames_processed': self.detection_stats['total_frames'],
                'total_detections': self.detection_stats['total_detections'],
                'drowning_detections': self.detection_stats['drowning_detections'],
                'other_detections': self.detection_stats['other_detections'],
                'drowning_events': self.detection_stats['drowning_events'],
                'start_time': self.detection_stats['start_time'].isoformat(),
                'end': self.detection_stats['end_time'].isoformat()
            },
            'detection_history': self.detection_stats['detection_history']
        }

        # 保存 JSON + CSV 报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(self.output_dir, f"detection_report_{timestamp}.json")
        csv_path = os.path.join(self.output_dir, f"detection_report_{timestamp}.csv")

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, ensure_ascii=False)
            self.logger.info(f"已保存JSON报告: {json_path}")
        except Exception as e:
            self.logger.error(f"保存JSON报告失败: {e}")
            return None

        try:
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['报告摘要'])
                writer.writerow(['处理时长(秒)', json_report['summary']['total_processing_time_seconds']])
                writer.writerow(['总帧数', json_report['summary']['total_frames_processed']])
                writer.writerow(['总检测数', json_report['summary']['total_detections']])
                writer.writerow(['溺水检测数', json_report['summary']['drowning_detections']])
                writer.writerow(['其他检测数', json_report['summary']['other_detections']])
                writer.writerow(['溺水事件数', json_report['summary']['drowning_events']])
                writer.writerow(['开始时间', json_report['summary']['start_time']])
                writer.writerow(['结束时间', json_report['summary']['end']])
                writer.writerow([])
                writer.writerow(['检测历史'])
                writer.writerow(['时间戳', '帧索引', '溺水数量', '总检测数', '检测详情'])
                for record in self.detection_stats['detection_history']:
                    detail = '; '.join(
                        f"{d['class_name']}({d['confidence']:.2f})" for d in record['detections'])
                    writer.writerow([record['timestamp'], record['frame_index'],
                                     record['drowning_count'], record['total_detections'], detail])
            self.logger.info(f"已保存CSV报告: {csv_path}")
        except Exception as e:
            self.logger.error(f"保存CSV报告失败: {e}")

        return json_path

    def process_frame(self, frame, frame_idx=None):
        """处理单帧：检测、绘制，返回处理后的帧和检测数量"""
        detections, drowning = self.detect(frame)

        # 更新统计数据
        self.update_detection_stats(detections, drowning, frame_idx)

        if drowning:
            self.logger.warning("检测到溺水者！")
            # 保存带检测框的溺水检测帧
            saved_path = self.save_drowning_frame(frame, detections, frame_idx)
            if self.alert_callback:
                self.alert_callback(frame, detections, saved_path)

        # 绘制检测框用于显示
        processed_frame = frame.copy()
        h, w = frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det['bbox']

            # 确保坐标是像素值
            if max(x1, y1, x2, y2) <= 1.0:
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h

            x1 = int(max(0, min(w, x1)))
            y1 = int(max(0, min(h, y1)))
            x2 = int(max(0, min(w, x2)))
            y2 = int(max(0, min(h, y2)))

            class_name = det['class_name']
            label = f"{class_name}: {det['confidence']:.2f}"
            color = (0, 0, 255) if class_name == 'Drowning' else (0, 255, 0)

            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(processed_frame, label, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 在帧上显示统计信息
        stats_text = f"Frames: {self.detection_stats['total_frames']} | Detections: {self.detection_stats['total_detections']} | Drowning: {self.detection_stats['drowning_detections']}"
        cv2.putText(processed_frame, stats_text, (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return processed_frame, len(detections)

    def run_on_video(self, video_source, output_path=None, display=True):
        """
        运行视频流处理，只保存检测到Drowning的帧
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            self.logger.error(f"无法打开视频源: {video_source}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.logger.info(f"开始处理视频源: {video_source}")
        self.logger.info("只保存检测到Drowning目标的帧（带检测框）")

        frame_idx = 0

        try:
            while True:
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理帧
                processed_frame, det_count = self.process_frame(frame, frame_idx)
                frame_idx += 1

                # 写入输出视频（如果需要）
                if out:
                    out.write(processed_frame)

                # 显示处理后的帧
                if display:
                    cv2.imshow('Drowning Detection', processed_frame)

                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            self.logger.info("用户中断处理")
        except Exception as e:
            self.logger.error(f"处理过程中发生错误: {e}")
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

            # 生成最终报告
            if self.output_dir:
                self.generate_report()

            self.logger.info(f"处理结束，总共检测到 {self.detection_stats['total_detections']} 个目标")
            self.logger.info(f"溺水检测: {self.detection_stats['drowning_detections']} 次")
            self.logger.info(f"溺水事件: {self.detection_stats['drowning_events']} 次")
            self.logger.info(f"已保存 {len(self.detection_stats['detection_history'])} 张溺水检测图片")


def default_alert_callback(frame, detections, saved_path=None):
    if saved_path:
        print(f"警报：检测到溺水！已保存带检测框图片: {saved_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drowning_alert_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"警报：检测到溺水！已保存图片: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="溺水检测子系统 (YOLOv8)")
    parser.add_argument('--model', type=str, required=True, help="模型文件路径")
    parser.add_argument('--source', type=str, default='0', help="视频源 (摄像头ID或视频文件路径)")
    parser.add_argument('--output', type=str, default=None, help="输出视频路径")
    parser.add_argument('--conf', type=float, default=0.5, help="置信度阈值")
    parser.add_argument('--no-display', action='store_true', help="不显示视频窗口")
    parser.add_argument('--imgsz', type=int, default=640, help="推理图像尺寸")
    parser.add_argument('--save-dir', type=str, default="./drowning_detections", help="保存溺水检测图片的目录路径")
    parser.add_argument('--output-dir', type=str, default="./reports", help="输出报告的目录路径")
    args = parser.parse_args()

    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    detector = DrowningDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        alert_callback=default_alert_callback,
        img_size=args.imgsz,
        save_dir=args.save_dir,
        output_dir=args.output_dir
    )

    detector.run_on_video(
        video_source=source,
        output_path=args.output,
        display=not args.no_display
    )
