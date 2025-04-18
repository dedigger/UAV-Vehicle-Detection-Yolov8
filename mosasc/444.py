import sys
import os
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QCheckBox, QSlider,
                             QComboBox, QGroupBox, QFormLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from ultralytics import YOLO
from paddleocr import PaddleOCR
import time
import random
from pathlib import Path
import tempfile

# Import DeepSORT
try:
    import deep_sort.deep_sort.deep_sort as ds

    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print("DeepSORT not available. Please install it before running this application.")
    sys.exit(1)


def putTextWithBackground(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,
                          text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)
    text_origin = (origin[0], origin[1] - 5)
    cv2.putText(img, text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_fps_signal = pyqtSignal(float)
    update_log_signal = pyqtSignal(str)
    update_detection_count_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.run_flag = True
        self.cap = None
        self.source = None
        self.yolo_model = None
        self.ocr = None
        self.deepsort_tracker = None
        self.enable_detection = True
        self.enable_tracking = True
        self.enable_ocr = True
        self.conf_thres = 0.25
        self.ocr_conf_thres = 0.7
        self.ocr_history = {}  # Track ID to OCR text mapping
        self.ocr_memory_frames = 30  # Keep OCR results for this many frames
        self.save_detections = False
        self.save_dir = None
        self.class_filter = None
        self.output_video = None
        self.recording = False
        self.output_path = None
        self.track_colors = {}  # Dictionary to store colors for each track ID

    def set_source(self, source):
        self.source = source
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(source)

        # Close any open output video
        if self.output_video is not None:
            self.output_video.release()
            self.output_video = None

    def load_models(self, yolo_weights, deepsort_weights=None):
        try:
            # Load YOLO model
            self.update_log_signal.emit("Loading YOLOv8 model...")
            self.yolo_model = YOLO(yolo_weights)
            self.update_log_signal.emit(f"YOLOv8 model loaded: {os.path.basename(yolo_weights)}")

            # Load OCR model
            self.update_log_signal.emit("Loading PaddleOCR model...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
            self.update_log_signal.emit("PaddleOCR model loaded")

            # Initialize DeepSORT
            if deepsort_weights:
                try:
                    self.deepsort_tracker = ds.DeepSort(deepsort_weights)
                    self.update_log_signal.emit("DeepSORT tracker initialized")
                except Exception as e:
                    self.update_log_signal.emit(f"DeepSORT initialization failed: {str(e)}")
                    return False
            else:
                self.update_log_signal.emit("DeepSORT weights not provided. Please select a model.")
                return False

            return True
        except Exception as e:
            self.update_log_signal.emit(f"Model loading failed: {str(e)}")
            return False

    def set_save_detections(self, enable, save_dir=None):
        self.save_detections = enable
        if enable and save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.update_log_signal.emit(f"Saving detections to: {save_dir}")
        else:
            self.save_detections = False

    def start_recording(self, output_path):
        if self.cap is None:
            self.update_log_signal.emit("No video source set")
            return False

        if self.output_video is not None:
            self.output_video.release()

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        if self.output_video.isOpened():
            self.recording = True
            self.update_log_signal.emit(f"Recording to: {output_path}")
            return True
        else:
            self.update_log_signal.emit(f"Failed to create output video file")
            return False

    def stop_recording(self):
        if self.output_video is not None:
            self.output_video.release()
            self.output_video = None
            self.recording = False
            self.update_log_signal.emit(f"Recording saved to: {self.output_path}")
            return True
        return False

    def set_class_filter(self, class_id):
        """Set a class ID filter (None for all classes)"""
        self.class_filter = class_id
        if class_id is not None:
            self.update_log_signal.emit(f"Filtering for class: {self.yolo_model.names[class_id]}")
        else:
            self.update_log_signal.emit("No class filter applied")

    def run(self):
        if self.cap is None or self.yolo_model is None or not self.run_flag:
            return

        frame_count = 0
        self.update_log_signal.emit("Starting video processing...")

        while self.run_flag:
            ret, frame = self.cap.read()
            if not ret:
                self.update_log_signal.emit("Video processing complete or unable to read frame")
                break

            start_time = time.time()
            processed_frame = self.process_frame(frame, frame_count)
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0

            # Add FPS information to the frame
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Record frame if recording is enabled
            if self.recording and self.output_video is not None:
                self.output_video.write(processed_frame)

            self.change_pixmap_signal.emit(processed_frame)
            self.update_fps_signal.emit(fps)

            frame_count += 1

    def process_frame(self, frame, frame_count):
        processed_frame = frame.copy()
        detection_count = 0

        # Step 1: YOLOv8 Detection
        if self.enable_detection:
            boxes, classes, confidences = self._perform_detection(frame)
            detection_count = len(boxes)
            self.update_detection_count_signal.emit(detection_count)

            # Save detected objects as separate images if enabled
            if self.save_detections and len(boxes) > 0:
                self._save_detection_crops(frame, boxes, classes, confidences, frame_count)

            # Step 2: Object Tracking with DeepSORT
            if self.enable_tracking and len(boxes) > 0 and self.deepsort_tracker:
                track_outputs = self._track_with_deepsort(frame, boxes, classes, confidences)

                # Process tracked objects
                for output in track_outputs:
                    x1, y1, x2, y2, track_id = output

                    # Find corresponding class ID
                    cls_id = None
                    for i, box in enumerate(boxes):
                        b_x1, b_y1, b_x2, b_y2 = box
                        if max(abs(x1 - b_x1), abs(y1 - b_y1), abs(x2 - b_x2),
                               abs(y2 - b_y2)) < 20:  # Approximate matching
                            cls_id = classes[i]
                            break

                    # Use default class if not found
                    if cls_id is None and classes:
                        cls_id = classes[0]
                    elif cls_id is None:
                        cls_id = 0

                    # Get or create a color for this track ID
                    if track_id not in self.track_colors:
                        self.track_colors[track_id] = (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255)
                        )
                    color = self.track_colors[track_id]

                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw bounding box
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)

                    # Display track information
                    class_name = self.yolo_model.names[cls_id]
                    conf = next((confidences[i] for i, box in enumerate(boxes)
                                 if max(abs(x1 - box[0]), abs(y1 - box[1]), abs(x2 - box[2]), abs(y2 - box[3])) < 20),
                                0)

                    # Use putTextWithBackground for better readability
                    putTextWithBackground(processed_frame,
                                          f"{class_name} {conf:.2f}",
                                          (x1, y1),
                                          font_scale=0.5,
                                          bg_color=(0, 0, 0))

                    putTextWithBackground(processed_frame,
                                          f"ID: {track_id}",
                                          (x1, y1 + 25),
                                          font_scale=0.5,
                                          bg_color=color)

                    # Step 3: OCR on tracked objects
                    if self.enable_ocr:
                        self._process_ocr(processed_frame, frame, track_id, x1, y1, x2, y2, color)

            # If tracking is disabled but detection is enabled
            elif not self.enable_tracking and len(boxes) > 0:
                self._process_detections_without_tracking(processed_frame, frame, boxes, classes, confidences)

        # Clean up old OCR history entries
        if frame_count % 50 == 0:
            self._cleanup_ocr_history()

        return processed_frame

    def _track_with_deepsort(self, frame, boxes, classes, confidences):
        """Convert boxes to DeepSORT format and use DeepSORT for tracking"""
        # DeepSORT expects detections in [x,y,w,h] format (center, width, height)
        # First convert [x1,y1,x2,y2] to [x,y,w,h]
        deepsort_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            deepsort_boxes.append([cx, cy, w, h])

        deepsort_boxes = np.array(deepsort_boxes)

        # Call DeepSORT update
        track_outputs = self.deepsort_tracker.update(deepsort_boxes, confidences, frame)

        # Convert back to [x1,y1,x2,y2,track_id] format for our use
        formatted_outputs = []
        for x1, y1, x2, y2, track_id in track_outputs:
            formatted_outputs.append([x1, y1, x2, y2, track_id])

        return formatted_outputs

    def _perform_detection(self, frame):
        results = self.yolo_model(frame, conf=self.conf_thres)

        boxes = []
        classes = []
        confidences = []

        for r in results:
            boxes_data = r.boxes
            for box in boxes_data:
                cls = int(box.cls[0].cpu().numpy())

                # Skip if not matching class filter (if one is set)
                if self.class_filter is not None and cls != self.class_filter:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()

                # Add margin to box
                h, w = frame.shape[:2]
                x1 = max(0, x1 - 5)
                y1 = max(0, y1 - 5)
                x2 = min(w, x2 + 5)
                y2 = min(h, y2 + 5)

                boxes.append([x1, y1, x2, y2])
                classes.append(cls)
                confidences.append(conf)

        return boxes, classes, confidences

    def _save_detection_crops(self, frame, boxes, classes, confidences, frame_count):
        """Save crops of detected objects as separate images"""
        if not self.save_dir:
            return

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = classes[i]
            conf = confidences[i]

            # Crop the region
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Create filename with class, frame, detection number and confidence
            class_name = self.yolo_model.names[cls_id]
            filename = f"{class_name}_frame{frame_count:05d}_{i}_conf{conf:.2f}.jpg"
            filepath = self.save_dir / filename

            # Save the image
            cv2.imwrite(str(filepath), crop)

    def _process_ocr(self, processed_frame, frame, track_id, x1, y1, x2, y2, color):
        # Check if we already have recent OCR results for this track
        if track_id in self.ocr_history and self.ocr_history[track_id]["age"] < self.ocr_memory_frames:
            # Use cached OCR result
            text = self.ocr_history[track_id]["text"]
            conf = self.ocr_history[track_id]["conf"]

            # Increase age counter
            self.ocr_history[track_id]["age"] += 1

            # Display OCR result
            if conf >= self.ocr_conf_thres:
                self._display_ocr_result(processed_frame, x1, y2, text, conf, color)
        else:
            # Extract the region of interest (ROI)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:  # Check if ROI is not empty
                try:
                    ocr_result = self.ocr.ocr(roi, cls=True)

                    # Process OCR results
                    if ocr_result and len(ocr_result) > 0 and len(ocr_result[0]) > 0:
                        best_text = ""
                        best_conf = 0

                        for line in ocr_result[0]:
                            if line and len(line) >= 2:
                                text = line[1][0]  # Recognized text
                                conf = line[1][1]  # Confidence score

                                # Keep highest confidence text
                                if conf > best_conf:
                                    best_text = text
                                    best_conf = conf

                        # Store OCR result in history
                        if best_text:
                            self.ocr_history[track_id] = {
                                "text": best_text,
                                "conf": best_conf,
                                "age": 0
                            }

                            # Display OCR result if confidence is high enough
                            if best_conf >= self.ocr_conf_thres:
                                self._display_ocr_result(processed_frame, x1, y2, best_text, best_conf, color)
                except Exception as e:
                    pass

    def _process_detections_without_tracking(self, processed_frame, frame, boxes, classes, confidences):
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = classes[i]
            conf = confidences[i]
            color = (0, 255, 0)  # Green for non-tracked objects

            # Draw bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)

            # Display detection information
            label = f"{self.yolo_model.names[cls_id]} {conf:.2f}"
            putTextWithBackground(processed_frame, label, (x1, y1), font_scale=0.5, bg_color=(0, 0, 0))

            # OCR on detected objects
            if self.enable_ocr:
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    try:
                        ocr_result = self.ocr.ocr(roi, cls=True)
                        if ocr_result and len(ocr_result) > 0 and len(ocr_result[0]) > 0:
                            for line in ocr_result[0]:
                                if line and len(line) >= 2:
                                    text = line[1][0]
                                    conf = line[1][1]
                                    if conf >= self.ocr_conf_thres:
                                        self._display_ocr_result(processed_frame, x1, y2, text, conf, color)
                    except Exception as e:
                        pass

    def _display_ocr_result(self, frame, x, y, text, conf, color):
        # Use the putTextWithBackground function for better readability
        putTextWithBackground(frame,
                              f"OCR: {text} ({conf:.2f})",
                              (x, y + 25),
                              font_scale=0.5,
                              text_color=(255, 255, 255),
                              bg_color=color)

    def _cleanup_ocr_history(self):
        # Remove old OCR history entries
        ids_to_delete = []
        for track_id, info in self.ocr_history.items():
            if info["age"] >= self.ocr_memory_frames:
                ids_to_delete.append(track_id)

        for track_id in ids_to_delete:
            del self.ocr_history[track_id]

    def stop(self):
        self.run_flag = False
        if self.cap is not None:
            self.cap.release()
        if self.output_video is not None:
            self.output_video.release()
        self.wait()

    def set_ocr_conf_threshold(self, value):
        self.ocr_conf_thres = value


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Object Detection, Tracking & OCR System")
        self.setGeometry(100, 100, 1200, 800)

        # Setup UI components
        self.setup_ui()

        # Setup video processing thread
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_fps_signal.connect(self.update_fps)
        self.video_thread.update_log_signal.connect(self.update_log)
        self.video_thread.update_detection_count_signal.connect(self.update_detection_count)

        # Initialize variables
        self.weights_path = None
        self.deepsort_weights_path = None
        self.video_path = None
        self.save_dir = None
        self.output_video_path = None

    def setup_ui(self):
        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Top control panel
        self.setup_control_panel()

        # Options panel
        self.setup_options_panel()

        # Status bar
        self.setup_status_bar()

        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(500)
        self.main_layout.addWidget(self.image_label, 1)

    def setup_control_panel(self):
        # Create model group
        model_group = QGroupBox("Models")
        model_layout = QFormLayout()

        # YOLO model selector
        self.btn_select_weights = QPushButton("Select YOLOv8 Model")
        self.btn_select_weights.clicked.connect(self.select_weights)
        model_layout.addRow("YOLOv8:", self.btn_select_weights)

        # DeepSORT model selector
        self.btn_select_deepsort = QPushButton("Select DeepSORT Model")
        self.btn_select_deepsort.clicked.connect(self.select_deepsort)
        model_layout.addRow("DeepSORT:", self.btn_select_deepsort)

        model_group.setLayout(model_layout)

        # Create source group
        source_group = QGroupBox("Source")
        source_layout = QFormLayout()

        # Video file selector
        self.btn_select_video = QPushButton("Select Video File")
        self.btn_select_video.clicked.connect(self.select_video)
        source_layout.addRow("Video:", self.btn_select_video)

        # Camera selector
        self.btn_use_camera = QPushButton("Use Camera")
        self.btn_use_camera.clicked.connect(self.use_camera)
        source_layout.addRow("Camera:", self.btn_use_camera)

        source_group.setLayout(source_layout)

        # Create control group
        control_group = QGroupBox("Controls")
        control_layout = QFormLayout()

        # Start/stop buttons
        self.btn_start = QPushButton("Start Processing")
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_start.setEnabled(False)
        control_layout.addRow("Process:", self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setEnabled(False)
        control_layout.addRow("Stop:", self.btn_stop)

        # Recording controls
        record_layout = QHBoxLayout()
        self.btn_start_recording = QPushButton("Start Recording")
        self.btn_start_recording.clicked.connect(self.start_recording)
        self.btn_start_recording.setEnabled(False)

        self.btn_stop_recording = QPushButton("Stop Recording")
        self.btn_stop_recording.clicked.connect(self.stop_recording)
        self.btn_stop_recording.setEnabled(False)

        record_layout.addWidget(self.btn_start_recording)
        record_layout.addWidget(self.btn_stop_recording)
        control_layout.addRow("Record:", record_layout)

        control_group.setLayout(control_layout)

        # Arrange the top control panel
        control_panel = QHBoxLayout()
        control_panel.addWidget(model_group)
        control_panel.addWidget(source_group)
        control_panel.addWidget(control_group)

        self.main_layout.addLayout(control_panel)

    def setup_options_panel(self):
        options_panel = QHBoxLayout()

        # Create detection group
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QFormLayout()

        # Confidence threshold slider
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setMinimum(10)
        self.slider_conf.setMaximum(95)
        self.slider_conf.setValue(25)
        self.slider_conf.valueChanged.connect(self.update_conf_threshold)
        detection_layout.addRow("Confidence Threshold:", self.slider_conf)

        # Confidence value label
        self.conf_value_label = QLabel("0.25")
        detection_layout.addRow("Value:", self.conf_value_label)

        # Class filter combo box
        self.class_filter_combo = QComboBox()
        self.class_filter_combo.addItem("All Classes", None)
        detection_layout.addRow("Class Filter:", self.class_filter_combo)
        self.class_filter_combo.currentIndexChanged.connect(self.update_class_filter)

        # Save detections checkbox
        self.chk_save_detections = QCheckBox("Save Detection Crops")
        self.chk_save_detections.stateChanged.connect(self.toggle_save_detections)
        detection_layout.addRow("Save:", self.chk_save_detections)

        # Browse save directory button
        self.btn_save_dir = QPushButton("Select Save Directory")
        self.btn_save_dir.clicked.connect(self.select_save_directory)
        detection_layout.addRow("Save Directory:", self.btn_save_dir)

        detection_group.setLayout(detection_layout)

        # Create OCR group
        ocr_group = QGroupBox("OCR Settings")
        ocr_layout = QFormLayout()

        # OCR confidence threshold slider
        self.slider_ocr_conf = QSlider(Qt.Horizontal)
        self.slider_ocr_conf.setMinimum(10)
        self.slider_ocr_conf.setMaximum(95)
        self.slider_ocr_conf.setValue(70)
        self.slider_ocr_conf.valueChanged.connect(self.update_ocr_conf_threshold)
        ocr_layout.addRow("OCR Confidence:", self.slider_ocr_conf)

        # OCR confidence value label
        self.ocr_conf_value_label = QLabel("0.70")
        ocr_layout.addRow("Value:", self.ocr_conf_value_label)

        ocr_group.setLayout(ocr_layout)

        # Create features group
        features_group = QGroupBox("Features")
        features_layout = QFormLayout()

        # Feature toggles
        self.chk_detection = QCheckBox("Enable Detection")
        self.chk_detection.setChecked(True)
        self.chk_detection.stateChanged.connect(self.toggle_detection)
        features_layout.addRow("Detection:", self.chk_detection)

        self.chk_tracking = QCheckBox("Enable Tracking")
        self.chk_tracking.setChecked(True)
        self.chk_tracking.stateChanged.connect(self.toggle_tracking)
        features_layout.addRow("Tracking:", self.chk_tracking)

        self.chk_ocr = QCheckBox("Enable OCR")
        self.chk_ocr.setChecked(True)
        self.chk_ocr.stateChanged.connect(self.toggle_ocr)
        features_layout.addRow("OCR:", self.chk_ocr)

        features_group.setLayout(features_layout)

        # Arrange the options panel
        options_panel.addWidget(detection_group)
        options_panel.addWidget(ocr_group)
        options_panel.addWidget(features_group)

        self.main_layout.addLayout(options_panel)

    def setup_status_bar(self):
        status_layout = QHBoxLayout()

        # FPS display
        self.fps_label = QLabel("FPS: 0.00")
        status_layout.addWidget(self.fps_label)

        # Detection count display
        self.detection_count_label = QLabel("Detections: 0")
        status_layout.addWidget(self.detection_count_label)

        # Log display
        self.log_label = QLabel("Ready")
        self.log_label.setWordWrap(True)
        status_layout.addWidget(self.log_label, 1)  # 1 = stretch factor

        self.main_layout.addLayout(status_layout)

    def select_weights(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select YOLOv8 Model", "",
                                                  "Model Files (*.pt *.pth);;All Files (*)")
        if filepath:
            self.weights_path = filepath
            self.btn_select_weights.setText(os.path.basename(filepath))
            self.update_start_button_state()

    def select_deepsort(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select DeepSORT Model", "",
                                                  "Model Files (*.t7);;All Files (*)")
        if filepath:
            self.deepsort_weights_path = filepath
            self.btn_select_deepsort.setText(os.path.basename(filepath))
            self.update_start_button_state()

    def select_video(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Video File", "",
                                                  "Video Files (*.mp4 *.avi *.mkv);;All Files (*)")
        if filepath:
            self.video_path = filepath
            self.btn_select_video.setText(os.path.basename(filepath))
            self.update_start_button_state()

    def use_camera(self):
        self.video_path = 0  # 0 is typically the default camera
        self.btn_use_camera.setText("Camera Selected")
        self.update_start_button_state()

    def select_save_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory for Saving Detections")
        if directory:
            self.save_dir = directory
            self.btn_save_dir.setText(os.path.basename(directory))

    def toggle_save_detections(self, state):
        is_checked = state == Qt.Checked
        if is_checked and not self.save_dir:
            self.select_save_directory()

        if self.video_thread:
            self.video_thread.set_save_detections(is_checked, self.save_dir)

    def update_conf_threshold(self):
        value = self.slider_conf.value() / 100.0
        self.conf_value_label.setText(f"{value:.2f}")
        if self.video_thread:
            self.video_thread.conf_thres = value

    def update_ocr_conf_threshold(self):
        value = self.slider_ocr_conf.value() / 100.0
        self.ocr_conf_value_label.setText(f"{value:.2f}")
        if self.video_thread:
            self.video_thread.set_ocr_conf_threshold(value)

    def update_class_filter(self):
        selected_data = self.class_filter_combo.currentData()
        if self.video_thread:
            self.video_thread.set_class_filter(selected_data)

    def toggle_detection(self, state):
        if self.video_thread:
            self.video_thread.enable_detection = state == Qt.Checked

    def toggle_tracking(self, state):
        if self.video_thread:
            self.video_thread.enable_tracking = state == Qt.Checked

    def toggle_ocr(self, state):
        if self.video_thread:
            self.video_thread.enable_ocr = state == Qt.Checked

    # Add this modified method to load_models in VideoThread class
    def load_models(self, yolo_weights, deepsort_weights=None):
        try:
            # Load YOLO model
            self.update_log_signal.emit("Loading YOLOv8 model...")
            self.yolo_model = YOLO(yolo_weights)
            self.update_log_signal.emit(f"YOLOv8 model loaded: {os.path.basename(yolo_weights)}")

            # Emit a signal with the model's class names for UI update
            class_names = self.yolo_model.names

            # Load OCR model
            self.update_log_signal.emit("Loading PaddleOCR model...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
            self.update_log_signal.emit("PaddleOCR model loaded")

            # Initialize DeepSORT
            if deepsort_weights:
                try:
                    self.deepsort_tracker = ds.DeepSort(deepsort_weights)
                    self.update_log_signal.emit("DeepSORT tracker initialized")
                except Exception as e:
                    self.update_log_signal.emit(f"DeepSORT initialization failed: {str(e)}")
                    return False
            else:
                self.update_log_signal.emit("DeepSORT weights not provided. Please select a model.")
                return False

            return True
        except Exception as e:
            self.update_log_signal.emit(f"Model loading failed: {str(e)}")
            return False

    # Modify start_processing method in MainWindow class
    def start_processing(self):
        if not self.weights_path or not self.video_path:
            self.update_log("Please select both a model and a video source.")
            return

        # Load models
        success = self.video_thread.load_models(self.weights_path, self.deepsort_weights_path)
        if not success:
            return

        # Set video source
        self.video_thread.set_source(self.video_path)

        # Update class filter dropdown with model classes
        self.update_class_dropdown()

        # Start processing
        self.video_thread.start()

        # Update UI state
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_start_recording.setEnabled(True)

        # Load models
        success = self.video_thread.load_models(self.weights_path, self.deepsort_weights_path)
        if not success:
            return

        # Set video source
        self.video_thread.set_source(self.video_path)

        # Update class filter dropdown with model classes
        self.update_class_dropdown()

        # Start processing
        self.video_thread.start()

        # Update UI state
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_start_recording.setEnabled(True)

    def update_class_dropdown(self):
        # Clear current items
        self.class_filter_combo.clear()
        self.class_filter_combo.addItem("All Classes", None)

        # Add classes from the model
        if self.video_thread.yolo_model:
            for idx, name in self.video_thread.yolo_model.names.items():
                self.class_filter_combo.addItem(name, idx)

    def stop_processing(self):
        self.video_thread.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_start_recording.setEnabled(False)
        self.btn_stop_recording.setEnabled(False)

    def start_recording(self):
        if not self.video_thread or not self.video_thread.isRunning():
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Output Video", "", "Video Files (*.avi);;All Files (*)"
        )

        if output_path:
            success = self.video_thread.start_recording(output_path)
            if success:
                self.btn_start_recording.setEnabled(False)
                self.btn_stop_recording.setEnabled(True)

    def stop_recording(self):
        success = self.video_thread.stop_recording()
        if success:
            self.btn_start_recording.setEnabled(True)
            self.btn_stop_recording.setEnabled(False)

    def update_start_button_state(self):
        # Enable start button only if both model and video are selected
        self.btn_start.setEnabled(bool(self.weights_path) and bool(self.video_path))

    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from OpenCV image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.2f}")

    def update_log(self, message):
        self.log_label.setText(message)

    def update_detection_count(self, count):
        self.detection_count_label.setText(f"Detections: {count}")

    def closeEvent(self, event):
        # Make sure video processing stops when the app is closed
        if self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()