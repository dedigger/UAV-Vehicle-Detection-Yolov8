import time
from pathlib import Path
import numpy as np
import cv2  # opencv-python
import matplotlib.pyplot as plt
from ultralytics import YOLO
import deep_sort.deep_sort.deep_sort as ds

# 全局变量，用于存储用户点击的点（定义计数线）
line_points = []

def draw_counting_line(event, x, y, flags, param):
    """
    鼠标回调函数：记录用户点击的点。
    """
    global line_points
    if event == cv2.EVENT_LBUTTONDOWN:
        line_points.append((x, y))
        if len(line_points) > 2:
            line_points = line_points[:2]  # 仅保留前两个点

def get_counting_line(frame):
    """
    显示视频第一帧，让用户点击确定计数线。
    点击两个点后，按 'q' 键结束选择，返回两个端点坐标。
    """
    global line_points
    line_points = []  # 清空已有的点

    temp_frame = frame.copy()
    cv2.namedWindow("Draw Counting Line")
    cv2.setMouseCallback("Draw Counting Line", draw_counting_line)

    instructions = ''
    while True:
        disp_frame = temp_frame.copy()
        # 绘制已选的点及连接线
        for pt in line_points:
            cv2.circle(disp_frame, pt, 5, (0, 255, 0), -1)
        if len(line_points) == 2:
            cv2.line(disp_frame, line_points[0], line_points[1], (0, 0, 255), 2)
        cv2.putText(disp_frame, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        cv2.imshow("Draw Counting Line", disp_frame)
        key = cv2.waitKey(1) & 0xFF
        # 当用户按 'q' 且已选取两个点时退出
        if key == ord('q') and len(line_points) == 2:
            break
    cv2.destroyWindow("Draw Counting Line")
    return tuple(line_points)

def putTextWithBackground(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,
                          text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    """
    在图像上绘制带背景色的文字，方便在不同背景上显示。
    """
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)
    text_origin = (origin[0], origin[1] - 5)
    cv2.putText(img, text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

def extract_detections(results, detect_classes):
    """
    检测目标的提取：返回检测框（以中心点+宽高描述）、置信度和类别
    :param results: YOLO 推理结果
    :param detect_classes: 检测类别，支持 int 或 list[int]
    :return: detections (N x 4 ndarray), confarray (list), class_ids (list)
    """
    detections = np.empty((0, 4))
    confarray = []
    class_ids = []
    for r in results:
        for box in r.boxes:
            cls_id = box.cls[0].int().item()
            if (isinstance(detect_classes, list) and cls_id in detect_classes) or \
               (isinstance(detect_classes, int) and cls_id == detect_classes):
                x_center, y_center, w, h = box.xywh[0].int().tolist()
                conf = round(box.conf[0].item(), 2)
                detections = np.vstack((detections, np.array([x_center, y_center, w, h])))
                confarray.append(conf)
                class_ids.append(cls_id)
    return detections, confarray, class_ids

def lines_intersect(A, B, C, D):
    """
    判断线段 AB 与 CD 是否相交
    :param A: 线段 AB 起点 (x, y)
    :param B: 线段 AB 终点 (x, y)
    :param C: 线段 CD 起点 (x, y)
    :param D: 线段 CD 终点 (x, y)
    :return: 如果相交返回 True，否则 False
    """
    def ccw(P, Q, R):
        return (R[1] - P[1]) * (Q[0] - P[0]) > (Q[1] - P[1]) * (R[0] - P[0])
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def detect_and_track(input_path: str, output_path: str, detect_classes, model, tracker, save_frame_dir=None) -> Path:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)
    output_video_path = Path(output_path) / "output.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter(output_video_path.as_posix(), fourcc, fps, size, isColor=True)

    # 读取第一帧，等待用户绘制计数线
    ret, first_frame = cap.read()
    if not ret:
        print("无法读取视频第一帧")
        return None

    # 显示第一帧，并等待用户通过鼠标点击确定计数线
    counting_line = get_counting_line(first_frame)
    print(f"计数线已确定，端点为: {counting_line}")

    # 为确保计数从视频开始，重置视频读取至第一帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 用于记录每个跟踪目标的上一次中心点，用于判断是否穿过计数线
    prev_centers = {}
    counted_ids = set()

    # 交通流率统计：每秒车辆通过数量及时间记录（用于后续可视化）
    count_in_interval = 0
    interval_start_time = time.time()
    flow_rate = 0
    flow_time = []       # 时间节点（单位：秒）
    flow_rate_list = []  # 对应流率

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用 YOLO 模型进行目标检测（stream 模式返回生成器结果）
        results = model(frame, stream=True)
        detections, confarray, class_ids = extract_detections(results, detect_classes)
        resultsTracker = tracker.update(detections, confarray, frame)

        # 可选：保存单独的检测目标图像
        if save_frame_dir and detections.shape[0] > 0:
            save_dir = Path(save_frame_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            for i, (x_center, y_center, w, h) in enumerate(detections):
                x_center, y_center, w, h = map(int, [x_center, y_center, w, h])
                x1 = max(0, x_center - w // 2)
                y1 = max(0, y_center - h // 2)
                x2 = min(frame.shape[1], x_center + w // 2)
                y2 = min(frame.shape[0], y_center + h // 2)
                obj_crop = frame[y1:y2, x1:x2]
                obj_name = model.names[class_ids[i]]
                frame_name = f"{obj_name}_frame{frame_idx:05d}_{i}.jpg"
                cv2.imwrite((save_dir / frame_name).as_posix(), obj_crop)

        # 在每一帧上绘制用户自定义的计数线
        cv2.line(frame, counting_line[0], counting_line[1], (0, 0, 255), 2)

        # 遍历所有跟踪目标，检测是否经过计数线
        for i, (x1, y1, x2, y2, track_id) in enumerate(resultsTracker):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            center = (int((x1+x2)/2), int((y1+y2)/2))

            # 绘制目标检测框及信息
            cls_id = class_ids[i] if i < len(class_ids) else -1
            cls_name = model.names[cls_id] if cls_id in model.names else "Unknown"
            conf = confarray[i] if i < len(confarray) else 0.0
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            putTextWithBackground(frame, label, (x1, y1), font_scale=0.8, bg_color=(0, 0, 0))
            putTextWithBackground(frame, f"ID: {track_id}", (x1, y1+30), font_scale=0.8, bg_color=(255, 0, 255))
            cv2.circle(frame, center, 3, (0, 255, 255), -1)

            # 判断目标是否经过计数线：若上一次中心点与本次中心点连线与计数线相交
            if track_id in prev_centers:
                prev_center = prev_centers[track_id]
                if lines_intersect(prev_center, center, counting_line[0], counting_line[1]):
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        count_in_interval += 1
            prev_centers[track_id] = center

        # 每隔 1 秒统计一次交通流率（车辆/秒）
        elapsed_time = time.time() - interval_start_time
        if elapsed_time >= 1.0:
            flow_rate = count_in_interval / elapsed_time
            flow_time.append(round(time.time(), 1))
            flow_rate_list.append(flow_rate)
            interval_start_time = time.time()
            count_in_interval = 0

        # 在画面上显示流率信息
        flow_text = f"Flow Rate: {flow_rate:.1f} veh/sec"
        cv2.putText(frame, flow_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        output_video.write(frame)
        cv2.imshow("Highway Traffic Flow Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户中断")
            break

        frame_idx += 1

    output_video.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f'输出视频保存在: {output_video_path}')

    # 视频处理完成后，绘制交通流率随时间变化的折线图
    if flow_time and flow_rate_list:
        plt.figure(figsize=(10,5))
        plt.plot(flow_time, flow_rate_list, marker='o', linestyle='-', color='b')
        plt.xlabel("Timestamp (s)")
        plt.ylabel("Flow Rate (veh/sec)")
        plt.title("Traffic Flow Rate Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return output_video_path

if __name__ == "__main__":
    # 视频文件路径（无人机静止拍摄的高速公路视频）
    input_path = r"D:\YOLO-World-master\YOLO-World-master\3333.mp4"
    output_path = r"D:\YOLO-World-master\YOLO-World-master"
    save_frame_dir = r"D:\YOLO-World-master\YOLO-World-master\saved_objs"

    # 加载YOLO模型（建议使用支持车辆检测的模型，如COCO预训练模型）
    model = YOLO(r"D:\YOLO-World-master\YOLO-World-master\yolov8n.pt")
    # 定义需要检测的车辆类别，例如：2: car, 3: motorcycle, 5: bus, 7: truck
    detect_classes = [2, 3, 5, 7]
    print("检测以下类别:")
    for cls in detect_classes:
        print(f"  {cls}: {model.names[cls]}")

    # 初始化 DeepSort 跟踪器（注意调整模块和权重文件路径）
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")

    detect_and_track(input_path, output_path, detect_classes, model, tracker, save_frame_dir)
