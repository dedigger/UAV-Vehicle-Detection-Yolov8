import tempfile
from pathlib import Path
import numpy as np
import cv2
import datetime
import re
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import deep_sort.deep_sort.deep_sort as ds
from paddleocr import PaddleOCR


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


def extract_detections(results, target_classes):
    """从模型结果中提取和处理检测信息。"""
    detections = np.empty((0, 5))  # x, y, w, h, class_id
    confarray = []

    for r in results:
        for box in r.boxes:
            cls_id = box.cls[0].int().item()
            if cls_id in target_classes:
                x1, y1, x2, y2 = box.xywh[0].int().tolist()
                conf = round(box.conf[0].item(), 2)
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, cls_id])))
                confarray.append(conf)

    return detections, confarray


def blur_objects(frame, detections):
    """对检测到的对象区域进行模糊处理。"""
    blurred_frame = frame.copy()
    for (x, y, w, h, _) in detections:
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(blurred_frame.shape[1], x2)
        y2 = min(blurred_frame.shape[0], y2)

        roi = blurred_frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        blurred = cv2.GaussianBlur(roi, (99, 99), 30)
        blurred_frame[y1:y2, x1:x2] = blurred

    return blurred_frame


def perform_ocr(frame, x1, y1, x2, y2, class_name, ocr_model, model_names):
    """对检测到的ROI区域进行OCR识别"""
    # 格式配置字典
    format_config = {
        'p xianzhisudu': {'prefix': '限速：', 'suffix': 'km/h'},
        'p xianzhigaodu': {'prefix': '限高：', 'suffix': 'm'},
        'p xianzhikuandu': {'prefix': '限宽：', 'suffix': 'm'},
        'p xianzhizhiliang': {'prefix': '限重：', 'suffix': 't'},
        'p xianzhizhouzhong': {'prefix': '限轴重：', 'suffix': 't'}
    }

    # 提取ROI区域
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    # PaddleOCR识别
    ocr_result = ocr_model.ocr(roi, cls=True)

    if ocr_result[0]:
        # 处理OCR结果
        texts = [line[1][0] for line in ocr_result[0]]
        full_text = ''.join(texts).lower()

        # 获取类别名称
        class_name = model_names[class_name]

        # 自动识别单位类型
        display_type = class_name
        if 'm' in full_text and 'gaodu' not in class_name and 'kuandu' not in class_name:
            display_type = 'p xianzhigaodu'  # 限高/限宽
        elif 't' in full_text and 'zhiliang' not in class_name and 'zhouzhong' not in class_name:
            display_type = 'p xianzhizhiliang'

        # 获取格式配置
        config = format_config.get(display_type, {})
        prefix = config.get('prefix', '')
        suffix = config.get('suffix', '')

        # 提取数值
        numbers = re.findall(r'\d+\.?\d*', full_text)
        value = numbers[0] if numbers else ''

        # 生成显示文本
        combined_text = f"{prefix}{value}{suffix}" if value else f"{prefix}识别失败"

        return combined_text

    return None


def detect_track_ocr(input_path, output_path, target_classes, detection_model, tracker, ocr_model):
    """处理视频，检测、跟踪目标，并进行OCR识别"""
    # 创建输出目录
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return None

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出视频
    output_video_path = output_dir / "output_combined.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # 加载中文字体
    try:
        font = ImageFont.truetype('simsun.ttc', 20)
    except:
        font = ImageFont.load_default()

    # 处理进度计数器
    frame_count = 0

    # 跟踪对象的历史OCR结果
    tracked_ocr_results = {}

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 进行目标检测
        results = detection_model(frame, stream=True)

        # 提取检测结果
        detections, confarray = extract_detections(results, target_classes)

        # 提取跟踪所需的检测框
        track_boxes = detections[:, :4] if len(detections) > 0 else np.empty((0, 4))

        # 更新跟踪器
        resultsTracker = tracker.update(track_boxes, confarray, frame)

        # 创建PIL图像用于绘制中文
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        # 处理跟踪结果
        for i, (x1, y1, x2, y2, track_id) in enumerate(resultsTracker):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # 只处理当前帧中有效的检测索引
            if i < len(detections):
                class_id = int(detections[i, 4])
                conf = confarray[i]

                # 绘制目标框
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 255), width=3)

                # 构建标签字符串
                class_name = detection_model.names[class_id]
                label = f"{class_name} {conf:.2f}"

                # 添加ID标签
                track_id_text = f"ID: {int(track_id)}"
                draw.text((x1, y1 - 25), label, fill=(255, 0, 255), font=font)
                draw.text((x1, y1), track_id_text, fill=(255, 0, 255), font=font)

                # 获取或执行OCR
                ocr_key = f"{track_id}_{class_id}"
                if ocr_key not in tracked_ocr_results:
                    ocr_text = perform_ocr(frame, x1, y1, x2, y2, class_id, ocr_model, detection_model.names)
                    if ocr_text:
                        tracked_ocr_results[ocr_key] = ocr_text

                # 如果有OCR结果，显示在图像上
                if ocr_key in tracked_ocr_results:
                    text_y = y2 + 25
                    draw.text((x1, text_y), tracked_ocr_results[ocr_key], fill=(255, 0, 0), font=font)

        # 转换回OpenCV格式
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 写入输出视频
        output_video.write(frame)

        # 显示处理进度
        frame_count += 1
        print(f"已处理帧数：{frame_count}/{total_frames} ({frame_count / total_frames:.1%})", end='\r')

        # 显示结果
        cv2.imshow("Detection, Tracking and OCR", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Stopped by user.")
            break
        elif key == ord('s'):
            # 按下 s 键，保存打码图像
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # 创建打码后的图像
            blurred_frame = blur_objects(frame.copy(), detections)

            # 保存原始图像和打码图像
            orig_filename = output_dir / f"original_frame_{timestamp}.jpg"
            blur_filename = output_dir / f"blurred_frame_{timestamp}.jpg"

            cv2.imwrite(str(orig_filename), frame)
            cv2.imwrite(str(blur_filename), blurred_frame)

            print(f"✅ Original frame saved at: {orig_filename}")
            print(f"✅ Blurred frame saved at: {blur_filename}")

            # 显示打码后的图像
            cv2.imshow("Blurred Image", blurred_frame)

    # 释放资源
    output_video.release()
    cap.release()
    cv2.destroyAllWindows()

    print(f'\n视频处理完成，输出视频保存在: {output_video_path}')
    return output_video_path


def initialize_models(yolo_path, target_classes):
    """初始化检测、跟踪和OCR模型"""
    # 加载YOLO模型
    detection_model = YOLO(yolo_path)

    # 初始化DeepSort跟踪器
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")

    # 初始化OCR模型
    ocr_model = PaddleOCR(use_angle_cls=True, lang='ch')

    return detection_model, tracker, ocr_model



if __name__ == "__main__":
    # 视频文件路径
    input_path = r"D:\YOLO-World-master\YOLO-World-master\3333.mp4"
    output_path = r"D:\YOLO-World-master\YOLO-World-master\videos"

    # 模型路径
    yolo_model_path = r"D:\YOLO-World-master\YOLO-World-master\yolov8n.pt"

    # 设置需要目标检测、跟踪和OCR识别的类别
    # 这里使用索引，根据实际模型类别进行调整
    target_classes = [0]  # 例如：限速、限高、限宽、限重、限轴重

    # 初始化模型
    detection_model, tracker, ocr_model = initialize_models(yolo_model_path, target_classes)

    # 执行检测、跟踪和OCR
    detect_track_ocr(input_path, output_path, target_classes, detection_model, tracker, ocr_model)