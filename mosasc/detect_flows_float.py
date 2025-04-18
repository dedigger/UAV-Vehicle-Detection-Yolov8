import tempfile
from pathlib import Path
import numpy as np
import cv2  # opencv-python
from ultralytics import YOLO

import deep_sort.deep_sort.deep_sort as ds


def putTextWithBackground(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,
                          text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)
    text_origin = (origin[0], origin[1] - 5)
    cv2.putText(img, text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)


def extract_detections(results, detect_class):
    detections = np.empty((0, 4))
    confarray = []
    for r in results:
        for box in r.boxes:
            if box.cls[0].int() == detect_class:
                # 获取检测框中心点坐标以及宽高信息（此处假设box.xywh返回[x_center, y_center, width, height]）
                # 注意：不同版本可能有不同格式，请根据实际情况调整
                x_center, y_center, width, height = box.xywh[0].int().tolist()
                conf = round(box.conf[0].item(), 2)
                # 如果需要转换为左上角和右下角，可自行计算，下面直接保存[center, width, height]格式
                detections = np.vstack((detections, np.array([x_center, y_center, width, height])))
                confarray.append(conf)
    return detections, confarray


def detect_and_track(input_path: str, output_path: str, detect_class: int, model, tracker, save_frame_dir=None) -> Path:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output_video_path = Path(output_path) / "output.avi"

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter(output_video_path.as_posix(), fourcc, fps, size, isColor=True)

    frame_idx = 0
    # 用于存储各跟踪目标上一次的中心位置和时间（秒）
    prev_positions = {}  # key: track ID, value: (center, time)
    # 用于累加逆向和同向目标计数
    total_opposite_count = 0  # X_a：逆向
    total_same_count = 0      # Y_c：顺向

    # 假设无人机飞行方向为右方，定义一个单位向量（实际可根据实际情况修改）
    drone_direction = np.array([1, 0])

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 进行目标检测（YOLO模型）
        results = model(frame, stream=True)
        detections, confarray = extract_detections(results, detect_class)
        resultsTracker = tracker.update(detections, confarray, frame)

        # 保存检测到的目标图像（可选）
        if save_frame_dir and detections.shape[0] > 0:
            save_dir = Path(save_frame_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            for i, det in enumerate(detections):
                # 这里将 det 格式为 [x_center, y_center, width, height] 转换为边界框
                x_center, y_center, width, height = map(int, det)
                x1 = max(0, x_center - width // 2)
                y1 = max(0, y_center - height // 2)
                x2 = min(frame.shape[1], x_center + width // 2)
                y2 = min(frame.shape[0], y_center + height // 2)
                obj_crop = frame[y1:y2, x1:x2]
                obj_name = model.names[detect_class]
                frame_name = f"{obj_name}_frame{frame_idx:05d}_{i}.jpg"
                cv2.imwrite((save_dir / frame_name).as_posix(), obj_crop)

        # 可视化跟踪结果，并更新流量计数（基于目标中心点运动方向）
        for i, track in enumerate(resultsTracker):
            # track 格式假设为 [x1, y1, x2, y2, Id]
            x1, y1, x2, y2, track_id = map(int, track)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            conf = confarray[i] if i < len(confarray) else 0.0
            label = f"{model.names[detect_class]} {conf:.2f}"
            putTextWithBackground(frame, label, (x1, y1), font_scale=0.8, bg_color=(0, 0, 0))
            putTextWithBackground(frame, f"ID: {int(track_id)}", (x1, y1 + 30), font_scale=0.8, bg_color=(255, 0, 255))

            # 计算目标当前中心点
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            current_time = frame_idx / fps  # 当前时间（秒）
            # 如果该目标曾在前一帧中被记录，则计算运动向量
            if track_id in prev_positions:
                prev_center, prev_time = prev_positions[track_id]
                dt = current_time - prev_time
                if dt > 0:
                    # 近似求解目标在图像中的位移向量
                    vel_vector = np.array(center) - np.array(prev_center)
                    # 根据与无人机运动方向对比判断，假设：若点积>0，说明目标运动方向与无人机一致（顺向），否则为逆向
                    dot = np.dot(vel_vector, drone_direction)
                    if dot > 0:
                        total_same_count += 1
                    elif dot < 0:
                        total_opposite_count += 1
                # 更新该目标的上一次位置和时间
                prev_positions[track_id] = (center, current_time)
            else:
                prev_positions[track_id] = (center, current_time)

        # 计算当前累计观察时间（秒）
        obs_time = (frame_idx + 1) / fps
        # 当前流率（辆/秒），再乘以 3600 得到辆/小时
        if obs_time > 0:
            current_flow_rate = ((total_opposite_count + total_same_count) / obs_time) * 3600/10000
        else:
            current_flow_rate = 0

        # 在图像上显示当前计算的流率
        putTextWithBackground(frame, f"Flow Rate: {int(current_flow_rate)} veh/hr", (30, 50), font_scale=1, bg_color=(50, 50, 50))

        output_video.write(frame)
        cv2.imshow("Detection and Tracking with Flow Rate", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user.")
            break

        frame_idx += 1

    output_video.release()
    cap.release()
    cv2.destroyAllWindows()

    # 最终总观察时间
    total_obs_time = frame_idx / fps
    final_flow_rate = ((total_opposite_count + total_same_count) / total_obs_time) * 3600/10000 if total_obs_time > 0 else 0
    print(f'Output video saved at: {output_video_path}')
    print(f'Estimated traffic flow rate: {int(final_flow_rate)} veh/hr')
    return output_video_path


if __name__ == "__main__":
    input_path = r"D:\YOLO-World-master\YOLO-World-master\3333.mp4"
    output_path = r"D:\YOLO-World-master\YOLO-World-master"
    save_frame_dir = r"D:\YOLO-World-master\YOLO-World-master\saved_objs"

    model = YOLO(r"D:\YOLO-World-master\YOLO-World-master\yolov8n (1).pt")
    detect_class = 0  # 例如，YOLO默认0可以是person或者车辆，根据实际情况修改
    print(f"Detecting: {model.names[detect_class]}")

    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    detect_and_track(input_path, output_path, detect_class, model, tracker, save_frame_dir)
