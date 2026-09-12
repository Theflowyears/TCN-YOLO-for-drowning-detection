import cv2
import json
import sys

def manual_bbox_annotation(video_path, start_frame, end_frame, output_json, keyframe_interval=10):
    print(f"[INFO] 开始标注，视频: {video_path}")
    print(f"[INFO] 帧范围: {start_frame} - {end_frame}, 间隔: {keyframe_interval} 帧")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] 无法打开视频")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] 视频总帧数: {total_frames}")

    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)
    if start_frame >= end_frame:
        print("[ERROR] 起始帧 >= 结束帧，退出")
        cap.release()
        return
    print(f"[INFO] 实际标注范围: {start_frame} - {end_frame}")

    keyframes = {}
    drawing = False
    ix, iy = -1, -1
    bbox_temp = []
    current_img = None  # 用于鼠标回调显示的图像

    def draw_rect(event, x, y, flags, param):
        nonlocal ix, iy, drawing, bbox_temp, current_img
        if current_img is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_copy = current_img.copy()
                cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('annotate', img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = min(ix, x), min(iy, y)
            x2, y2 = max(ix, x), max(iy, y)
            bbox_temp = [x1, y1, x2, y2]
            print(f"  选定 bbox: {bbox_temp}")

    cv2.namedWindow('annotate')
    cv2.setMouseCallback('annotate', draw_rect)

    keyframe_candidates = [f for f in range(start_frame, end_frame + 1)
                           if f == start_frame or f == end_frame or f % keyframe_interval == 0]
    print(f"[INFO] 需要手动标注的关键帧号: {keyframe_candidates}")

    for fid in keyframe_candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] 无法读取帧 {fid}，跳过")
            continue
        print(f"\n[ANNOTATE] 正在标注帧 {fid}")
        bbox_temp = []
        current_img = frame.copy()
        cv2.imshow('annotate', current_img)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 13 and bbox_temp:  # Enter
                keyframes[fid] = bbox_temp
                print(f"  -> 已保存 {fid} 的 bbox: {bbox_temp}")
                break
            elif key == ord('s'):
                print(f"  -> 跳过帧 {fid}")
                break
            elif key == ord('q'):
                print("[EXIT] 用户退出")
                cap.release()
                cv2.destroyAllWindows()
                return
            # 其他按键忽略，保持窗口显示
            cv2.imshow('annotate', current_img)

    cap.release()
    cv2.destroyAllWindows()

    if len(keyframes) < 2:
        print("[ERROR] 至少需要标注两个关键帧才能插值，已退出")
        return

    print(f"[INFO] 关键帧标注完成，共 {len(keyframes)} 帧，开始插值...")

    sorted_fids = sorted(keyframes.keys())
    all_frames = list(range(start_frame, end_frame + 1))
    all_bboxes = []
    for fid in all_frames:
        if fid in keyframes:
            all_bboxes.append(keyframes[fid])
        else:
            prev_f = max([k for k in sorted_fids if k < fid], default=None)
            next_f = min([k for k in sorted_fids if k > fid], default=None)
            if prev_f is None or next_f is None:
                near = prev_f if prev_f is not None else next_f
                all_bboxes.append(keyframes[near])
            else:
                alpha = (fid - prev_f) / (next_f - prev_f)
                bbox_interp = [
                    keyframes[prev_f][i] + (keyframes[next_f][i] - keyframes[prev_f][i]) * alpha
                    for i in range(4)
                ]
                all_bboxes.append(bbox_interp)

    track = {
        "track_id": 0,
        "video_name": video_path.split("\\")[-1],
        "frames": all_frames,
        "bboxes": all_bboxes,
        "confs": [1.0] * len(all_frames),
        "label": 1
    }
    with open(output_json, 'w') as f:
        json.dump([track], f, indent=2)
    print(f"[DONE] 溺水轨迹已保存到 {output_json}，共 {len(all_frames)} 帧")
    print(f"关键帧坐标: {keyframes}")

if __name__ == "__main__":
    print(f"命令行参数: {sys.argv}")
    if len(sys.argv) < 4:
        print("用法: python manual_label_drowning.py <视频路径> <起始帧> <结束帧> [输出文件] [--keyframe_interval 间隔]")
        sys.exit(1)
    video = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    output = "drowning_track.json"
    interval = 10
    idx = 4
    while idx < len(sys.argv):
        if sys.argv[idx] == "--keyframe_interval" and idx + 1 < len(sys.argv):
            interval = int(sys.argv[idx + 1])
            idx += 2
        else:
            output = sys.argv[idx]
            idx += 1
    print(f"解析后参数: 视频={video}, 起始={start}, 结束={end}, 输出={output}, 间隔={interval}")
    manual_bbox_annotation(video, start, end, output, interval)