import cv2
import json
import os
import sys

def label_tracks(video_path, json_path, output_json):
    # 读取轨迹
    with open(json_path, 'r') as f:
        tracks = json.load(f)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频 {video_path}")
        return

    # 为每条轨迹增加 label 字段（如果还没标）
    for track in tracks:
        if 'label' not in track:
            track['label'] = -1  # 未标注

    # 需要标注的轨迹列表
    to_label = [t for t in tracks if t['label'] == -1]
    if not to_label:
        print("所有轨迹都已标注！")
        cap.release()
        return

    total = len(to_label)
    print(f"共有 {total} 条轨迹需要标注。")
    print("操作说明：")
    print("  按 '0' → 标记为正常")
    print("  按 '1' → 标记为溺水")
    print("  按 's' → 跳过这条轨迹（保留未标注）")
    print("  按 'q' → 退出标注")

    for idx, track in enumerate(to_label):
        tid = track['track_id']
        frames = track['frames']
        bboxes = track['bboxes']

        print(f"\n正在标注 track {tid} （第 {idx+1}/{total} 个）")
        print("播放该轨迹的裁剪片段...（按空格暂停/继续）")

        # 播放轨迹片段
        pause = False
        for i, fid in enumerate(frames):
            if pause:
                key = cv2.waitKey(0) & 0xFF
                if key == ord(' '):  # 空格继续
                    pause = False
                    continue
                elif key == ord('q'):
                    cap.release()
                    return
            else:
                key = cv2.waitKey(30) & 0xFF  # 30ms一帧

            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret:
                print(f"  读取帧 {fid} 失败，跳过")
                continue

            # 裁剪 bbox 区域
            x1, y1, x2, y2 = map(int, bboxes[i])
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # 放大显示
            roi = cv2.resize(roi, (224, 224))
            cv2.putText(roi, f"Track {tid}  Frame {fid}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow('Track Player', roi)

            if key == ord(' '):
                pause = True
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('0'):
                track['label'] = 0
                print("  -> 标记为正常 (0)")
                break
            elif key == ord('1'):
                track['label'] = 1
                print("  -> 标记为溺水 (1)")
                break
            elif key == ord('s'):
                print("  -> 跳过")
                break

        # 如果播放完也没有按键，默认跳过
        if track['label'] == -1:
            print("  未按键，已跳过（仍为未标注）")
        else:
            print(f"  已标注 track {tid} 为 {track['label']}")

        # 每标注一条就保存一次，防止意外丢失
        with open(output_json, 'w') as f:
            json.dump(tracks, f, indent=2)

    cap.release()
    cv2.destroyAllWindows()

    # 统计
    labeled = [t for t in tracks if t['label'] != -1]
    unlabeled = [t for t in tracks if t['label'] == -1]
    print(f"\n标注完成！")
    print(f"已标注: {len(labeled)} 条 (正常: {sum(1 for t in labeled if t['label']==0)}, 溺水: {sum(1 for t in labeled if t['label']==1)})")
    print(f"未标注: {len(unlabeled)} 条")
    print(f"结果已保存至 {output_json}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python label_tracks.py <视频路径> <轨迹JSON路径>")
        sys.exit(1)

    video_file = sys.argv[1]
    json_file = sys.argv[2]
    # 输出为原文件名_labeled.json
    out_file = os.path.splitext(json_file)[0] + "_labeled.json"
    label_tracks(video_file, json_file, out_file)