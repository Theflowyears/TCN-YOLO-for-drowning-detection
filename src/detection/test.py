import threading
import cv2
import argparse
import logging
from datetime import datetime
import requests
from ultralytics import YOLO

class DrowningDetector:
    def __init__(self, model_path, conf_threshold=0.7, alert_callback=None,img_size=1280,api_url=None, enable_api=False):
        """
        :param model_path: YOLO模型文件路径 (例如 best.pt)
        :param conf_threshold: 置信度阈值
        :param alert_callback: 警报回调函数，接收 (frame, detections) 参数
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.img_size = img_size          # --imgsz 之前被解析后直接丢弃，这里真正接上
        self.alert_callback = alert_callback
        self.logger = self._setup_logger()
        # 类别名称直接从模型获取，避免硬编码
        self.class_names = self.model.names  # dict {0: 'swimming', 1: 'drowning', ...}
        self.api_url = api_url
        self.enable_api = enable_api
        if self.enable_api and not self.api_url:
            self.logger.warning("API功能已启用但未提供API URL，将不会发送请求")
            self.enable_api = False

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
        """
        对单帧进行目标检测
        :return: (detections_list, drowning_detected)
        """
        results = self.model(frame, conf=self.conf_threshold, imgsz=self.img_size)[0]
        detections = []
        drowning_detected = False

        if results.boxes is None:
            return detections, drowning_detected

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

            # 类别名按大小写不敏感比较：数据集里是 'Drowning'，早期代码按 'drowning' 匹配，
            # 严格相等会让告警永远不触发。
            if class_name.lower() == 'drowning' and conf >= self.conf_threshold:
                drowning_detected = True

        return detections, drowning_detected

    def _send_api_alert(self, frame, detections):
        """异步向云端推送遇险信号（未启用 API 时直接返回）。

        注意：该方法必须属于 DrowningDetector —— 原快照里它被写在模块顶层
        （位于 `if __name__ == "__main__"` 之后），`self._send_api_alert(...)`
        会直接抛 AttributeError。
        """
        if not self.enable_api or not self.api_url:
            return

        drowning_targets = [d for d in detections if d['class_name'].lower() == 'drowning']
        if not drowning_targets:
            return

        payload = {
            'timestamp': datetime.now().isoformat(),
            'drowning_count': len(drowning_targets),
            'detections': [
                {
                    'bbox': d['bbox'],
                    'confidence': d['confidence']
                } for d in drowning_targets
            ]
        }

        def send():
            try:
                response = requests.post(self.api_url, json=payload, timeout=5)
                if response.status_code >= 400:
                    self.logger.error(f"API请求失败: {response.status_code} {response.text}")
                else:
                    self.logger.info("遇险信号已发送至云服务器")
            except Exception as e:
                self.logger.error(f"发送API请求时发生异常: {e}")

        threading.Thread(target=send, daemon=True).start()

    def process_frame(self, frame):
        # 1. 执行检测
        detections, drowning = self.detect(frame)

        # 2. 警报处理
        if drowning:
            self.logger.warning("检测到溺水者！")
            if self.alert_callback:
                self.alert_callback(frame, detections)
            self._send_api_alert(frame, detections)

        h, w = frame.shape[:2]

        # 4. 绘制所有检测框
        for det in detections:
            x1, y1, x2, y2 = det['bbox']

            if max(x1, y1, x2, y2) <= 1.0:
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h

            x1 = int(max(0, min(w, x1)))
            y1 = int(max(0, min(h, y1)))
            x2 = int(max(0, min(w, x2)))
            y2 = int(max(0, min(h, y2)))

            class_name = det['class_name']
            label = f"{class_name}: {det['confidence']:.2f}"

            if class_name.lower() == 'drowning':
                color = (0, 0, 255)  # BGR红色
            else:
                color = (0, 255, 0)  # BGR绿色

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(frame, label, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, len(detections)

    def run_on_video(self, video_source, output_path=None, display=True, frame_count=0):
        """
        运行视频流处理
        :param video_source: 摄像头索引（如0）或视频文件路径
        :param output_path: 输出视频文件保存路径（可选）
        :param display: 是否显示实时窗口
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            self.logger.error(f"无法打开视频源: {video_source}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.logger.info(f"开始处理视频源: {video_source}")

        total_detections = 0
        paused = False  # 新增：暂停标志
        current_frame = None  # 新增：保存当前帧，用于暂停时显示

        try:
            while True:

                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    current_frame = frame.copy()  # 保存当前帧，供暂停时显示
                    # 执行检测和绘制
                    processed_frame, det_count = self.process_frame(current_frame)
                    total_detections += det_count
                    # 在画面上显示运行状态
                    cv2.putText(processed_frame, "RUNNING", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    if current_frame is None:
                        continue
                    processed_frame = current_frame.copy()
                    # 在画面上显示暂停状态
                    cv2.putText(processed_frame, "PAUSED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if display:
                    cv2.imshow('Drowning Detection', processed_frame)

                if out and not paused:
                    out.write(processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    if paused:
                        self.logger.info("识别已暂停")
                    else:
                        self.logger.info("识别继续")

        except KeyboardInterrupt:
            self.logger.info("用户中断处理")
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            self.logger.info(f"处理结束，总共检测到 {total_detections} 个目标")


def default_alert_callback(frame, detections):
    """
    默认警报函数：保存当前帧为图片
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"drowning_alert_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"警报：溺水者！已保存图片 {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="溺水检测子系统 (YOLOv8)")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--api-url', type=str, default=None,
                        help='云端告警接收接口，需配合 --enable-api')
    parser.add_argument('--enable-api', action='store_true',
                        help='检测到溺水时向 --api-url 异步 POST 告警')
    args = parser.parse_args()

    # 处理摄像头索引
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    detector = DrowningDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        img_size=args.imgsz,
        alert_callback=default_alert_callback,
        api_url=args.api_url,
        enable_api=args.enable_api
    )

    detector.run_on_video(
        video_source=source,
        output_path=args.output,
        display=not args.no_display
    )