import tempfile
from pathlib import Path
import numpy as np
import cv2  # opencv-python
from ultralytics import YOLO
import deep_sort.deep_sort.deep_sort as ds
import datetime


def putTextWithBackground(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX,
                          font_scale=1, text_color=(255, 255, 255),
                          bg_color=(0, 0, 0), thickness=1):
    """绘制带有背景的文本。"""
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)
    text_origin = (origin[0], origin[1] - 5)
    cv2.putText(img, text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)


def extract_detections(results, detect_class):
    """从模型结果中提取和处理检测信息。"""
    detections = np.empty((0, 4))
    confarray = []

    for r in results:
        for box in r.boxes:
            if box.cls[0].int() == detect_class:
                x1, y1, x2, y2 = box.xywh[0].int().tolist()
                conf = round(box.conf[0].item(), 2)
                detections = np.vstack((detections, np.array([x1, y1, x2, y2])))
                confarray.append(conf)
    return detections, confarray


def blur_objects(frame, detections):
    """对检测到的对象区域进行模糊处理。"""
    for (x, y, w, h) in detections:
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        blurred = cv2.GaussianBlur(roi, (99, 99), 30)
        frame[y1:y2, x1:x2] = blurred
    return frame


def detect_and_track(input_path: str, output_path: str, detect_class: int, model, tracker) -> Path:
    """处理视频，检测并跟踪目标。"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output_video_path = Path(output_path) / "output.avi"

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter(output_video_path.as_posix(), fourcc, fps, size, isColor=True)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, stream=True)
        detections, confarray = extract_detections(results, detect_class)
        resultsTracker = tracker.update(detections, confarray, frame)

        for i, (x1, y1, x2, y2, Id) in enumerate(resultsTracker):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # 绘制目标框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # 构建标签字符串：类别 + 置信度
            conf = confarray[i] if i < len(confarray) else 0.0
            label = f"{model.names[detect_class]} {conf:.2f}"

            # 绘制类别与ID文本
            putTextWithBackground(frame, label, (x1, y1), font_scale=0.8, bg_color=(0, 0, 0))
            putTextWithBackground(frame, f"ID: {int(Id)}", (x1, y1 + 30), font_scale=0.8, bg_color=(255, 0, 255))

        output_video.write(frame)
        cv2.imshow("Detection and Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Stopped by user.")
            break
        elif key == ord('s'):
            # 按下 s 键，保存打码图像
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_frame = frame.copy()
            blurred_frame = blur_objects(save_frame, detections)
            filename = Path(output_path) / f"saved_frame_{timestamp}.jpg"
            cv2.imwrite(str(filename), blurred_frame)
            print(f"✅ Frame saved and blurred at: {filename}")

    output_video.release()
    cap.release()
    cv2.destroyAllWindows()

    print(f'Output video saved at: {output_video_path}')
    return output_video_path


if __name__ == "__main__":
    input_path = r"D:\YOLO-World-master\YOLO-World-master\3333.mp4"
    output_path = r"D:\YOLO-World-master\YOLO-World-master"

    # 加载模型
    model = YOLO(r"D:\YOLO-World-master\YOLO-World-master\yolov8n (1).pt")
    detect_class = 0  # 'person'
    print(f"Detecting: {model.names[detect_class]}")

    # DeepSort 初始化
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")

    detect_and_track(input_path, output_path, detect_class, model, tracker)
