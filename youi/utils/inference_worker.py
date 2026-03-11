import os
import sys
import time
import traceback
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import cv2
import torch

class InferenceWorker(QObject):
    """Worker for running YOLO model inference in a separate thread."""
    
    # Signals
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    stats_update = pyqtSignal(str)
    image_update = pyqtSignal(str)
    inference_complete = pyqtSignal()
    inference_error = pyqtSignal(str)

    def __init__(self, model_path, input_path, output_dir, is_folder_mode=False,
                 is_video_mode=False, is_camera_mode=False,
                 conf_thresh=0.25, iou_thresh=0.45, img_size=640, save_results=True):
        super().__init__()
        self.model_path = model_path
        self.input_path = input_path
        self.output_dir = output_dir
        self.is_folder_mode = is_folder_mode
        self.is_video_mode = is_video_mode
        self.is_camera_mode = is_camera_mode
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        self.save_results = save_results
        self.should_stop = False
        self.model = None

    def stop(self):
        """Signal the worker to stop processing."""
        self.should_stop = True
        self.log_update.emit("接收到停止信号，正在停止...")
    
    def run(self):
        """Execute the inference process."""
        try:
            # 开始使用当前模型进行识别
            self.log_update.emit(f"Начало распознавания с использованием модели: {os.path.basename(self.model_path)}")

            # 当前模式：文件夹模式 / 图片模式
            self.log_update.emit(f"Режим: {'Режим папки' if self.is_folder_mode else 'Режим изображения'}")

            # 输入路径
            self.log_update.emit(f"Входной путь: {self.input_path}")

            # 输出目录
            self.log_update.emit(f"Выходной каталог: {self.output_dir}")

            # 置信度阈值
            self.log_update.emit(f"Порог уверенности: {self.conf_thresh}")

            # IoU 阈值
            self.log_update.emit(f"Порог IoU: {self.iou_thresh}")

            # 图像尺寸
            self.log_update.emit(f"Размер изображения: {self.img_size}")
            
            # Make sure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Load YOLO model
            self.load_model()
            
            if self.should_stop:
                # 推理已停止
                self.log_update.emit("Распознавание остановлено")
                return
            
            # Run inference based on mode
            # Run inference based on mode
            if self.is_camera_mode:
                self.run_camera_inference()
            elif self.is_video_mode:
                self.run_video_inference()
            elif self.is_folder_mode:
                self.run_folder_inference()
            else:
                if ';' in self.input_path:
                    self.run_multiple_images_inference()
                else:
                    self.run_single_image_inference()

            if not self.should_stop:
                self.inference_complete.emit()
                
        except Exception as e:
            error_msg = f"Ошибка в процессе распознавания: {str(e)}\n{traceback.format_exc()}"
            self.inference_error.emit(error_msg)

    def run_video_inference(self):
        """Run inference on a video file with overall speed measurement."""
        self.log_update.emit("Выполняется распознавание видео...")

        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видеофайл: {self.input_path}")

        results_dir = os.path.join(self.output_dir, 'inference_results')
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, 'video_output.avi')

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # 原视频FPS，仅用于保存输出视频，不代表实际处理速度
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps is None or source_fps <= 0:
            source_fps = 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, source_fps, (width, height))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed = 0

        # 总体测速：包含读帧、推理、plot、写视频等完整流程
        total_start_time = time.perf_counter()
        inference_total_time = 0.0

        try:
            while cap.isOpened() and not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    break

                infer_start = time.perf_counter()
                results = self.model(
                    frame,
                    imgsz=self.img_size,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh
                )
                inference_total_time += time.perf_counter() - infer_start

                result_img = results[0].plot()
                out.write(result_img)

                processed += 1

                if frame_count > 0:
                    progress = int((processed / frame_count) * 100)
                    self.progress_update.emit(progress)

                    if processed % max(1, frame_count // 10) == 0:
                        self.image_update.emit(output_path)
        finally:
            total_end_time = time.perf_counter()
            cap.release()
            out.release()

        self.image_update.emit(output_path)

        total_time = total_end_time - total_start_time
        avg_total_time = total_time / processed if processed > 0 else 0.0
        avg_infer_time = inference_total_time / processed if processed > 0 else 0.0
        actual_fps = processed / total_time if total_time > 0 else 0.0

        self.log_update.emit(f"Обработка видео завершена, путь сохранения: {output_path}")

        self.stats_update.emit(
            "Количество обработанных кадров: {processed}\n"
            "Общее время обработки: {total_time:.2f} с\n"
            "Среднее время на кадр (полный цикл): {avg_total_time:.4f} с\n"
            "Среднее время инференса на кадр: {avg_infer_time:.4f} с\n"
            "Средняя скорость обработки: {actual_fps:.2f} FPS".format(
                processed=processed,
                total_time=total_time,
                avg_total_time=avg_total_time,
                avg_infer_time=avg_infer_time,
                actual_fps=actual_fps
            )
        )

        self.progress_update.emit(100)

    def run_camera_inference(self):
        """Run inference from live camera with overall speed measurement."""
        self.log_update.emit("Выполняется распознавание в реальном времени с камеры...")

        try:
            cam_index = int(self.input_path.strip())
        except ValueError:
            raise ValueError(f"Неверный индекс камеры: {self.input_path}")

        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть камеру {cam_index}")

        results_dir = os.path.join(self.output_dir, 'inference_results')
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, 'camera_output.avi')

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # 这里只是保存输出视频时使用的FPS，不代表实际处理速度
        save_fps = 20

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, save_fps, (width, height))

        frame_id = 0

        # 总体测速：包含采集、推理、plot、写视频等完整流程
        total_start_time = time.perf_counter()
        inference_total_time = 0.0

        try:
            while cap.isOpened() and not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    break

                infer_start = time.perf_counter()
                results = self.model(
                    frame,
                    imgsz=self.img_size,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh
                )
                inference_total_time += time.perf_counter() - infer_start

                result_img = results[0].plot()
                out.write(result_img)

                frame_id += 1

                if frame_id % 10 == 0:
                    self.image_update.emit(output_path)
                    self.progress_update.emit(min(100, frame_id))  # 保留你原来的逻辑
        finally:
            total_end_time = time.perf_counter()
            cap.release()
            out.release()

        self.image_update.emit(output_path)

        total_time = total_end_time - total_start_time
        avg_total_time = total_time / frame_id if frame_id > 0 else 0.0
        avg_infer_time = inference_total_time / frame_id if frame_id > 0 else 0.0
        actual_fps = frame_id / total_time if total_time > 0 else 0.0

        self.log_update.emit(
            f"Распознавание с камеры в реальном времени завершено, путь сохранения: {output_path}"
        )

        self.stats_update.emit(
            "Общее количество кадров: {frame_id}\n"
            "Общее время обработки: {total_time:.2f} с\n"
            "Среднее время на кадр (полный цикл): {avg_total_time:.4f} с\n"
            "Среднее время инференса на кадр: {avg_infer_time:.4f} с\n"
            "Средняя скорость обработки: {actual_fps:.2f} FPS".format(
                frame_id=frame_id,
                total_time=total_time,
                avg_total_time=avg_total_time,
                avg_infer_time=avg_infer_time,
                actual_fps=actual_fps
            )
        )

        self.progress_update.emit(100)

    def load_model(self):
        """Load the YOLO model."""
        self.log_update.emit("Загрузка модели YOLO...")
        
        try:
            # Try to use Ultralytics YOLO
            import ultralytics
            from ultralytics import YOLO
            
           # 输出当前使用的 Ultralytics 版本
            self.log_update.emit(f"Используется версия Ultralytics: {ultralytics.__version__}")

            # 检查当前是否为 yolo12 模型
            is_yolo12 = 'yolo12' in os.path.basename(self.model_path).lower()

            # 根据模型类型使用合适的参数加载模型
            if is_yolo12:
                # 检测到 yolo12，使用兼容模式加载
                self.log_update.emit("Обнаружена модель yolo12, используется режим совместимости для загрузки")
                self.model = YOLO(self.model_path, task='detect')
            else:
                # 普通方式加载模型
                self.model = YOLO(self.model_path)

            # 输出模型信息
            self.log_update.emit(f"Модель загружена: {self.model_path}")
            self.log_update.emit(f"Задача модели: {self.model.task}")
            self.log_update.emit("Модель успешно загружена")

            # 检查模型属性以判断兼容的 API 版本
            self.ultralytics_version = "v8+" if hasattr(self.model, 'predict') else "v5"
            self.log_update.emit(f"Обнаружена совместимость API: Ultralytics {self.ultralytics_version}")

        except ImportError:
            # 未找到 Ultralytics 包时，尝试使用 torch hub
            self.log_update.emit("Пакет Ultralytics не найден, выполняется попытка использования torch hub...")
            try:
                # 回退到 torch hub 加载 YOLOv5
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
                self.model.conf = self.conf_thresh
                self.model.iou = self.iou_thresh
                self.model.classes = None  # 所有类别
                self.model.max_det = 300   # 最大检测数量
                self.ultralytics_version = "torch_hub"
                
                # 输出加载成功日志
                self.log_update.emit("Модель загружена через torch hub")
            except Exception as e:
                # 模型加载失败时抛出异常
                raise ValueError(f"Не удалось загрузить модель: {str(e)}")
    
    def run_single_image_inference(self):
        """Run inference on a single image."""
        self.log_update.emit("Распознавание одного изображения...")
        
        # Get image path
        img_path = self.input_path
        
        # Set up output directory
        results_dir = os.path.join(self.output_dir, 'inference_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Update progress
        self.progress_update.emit(10)  # 10% - Starting
        
        # Run inference
        try:
            self.log_update.emit(f"Распознать изображение: {os.path.basename(img_path)}")
            
            # Check if this is a yolo12 model
            is_yolo12 = 'yolo12' in os.path.basename(self.model_path).lower()
            
            # Try to use the appropriate API
            if is_yolo12:
                # yolo12 API requires specific parameters
                self.log_update.emit("Использовать API YOLO12 для распознавания")
                results = self.model(img_path, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
            else:
                # Try to use the newer Ultralytics v8 API first
                try:
                    results = self.model(img_path, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                except TypeError as e:
                    # Log error and try with compatible parameters
                    self.log_update.emit(f"Ошибка параметров API: {str(e)}, попытка использовать обратно совместимые параметры")
                    # Use older parameter name (size)
                    results = self.model(img_path, size=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                
            # Update progress
            self.progress_update.emit(50)  # 50% - Inference done
            
            # Count detections
            detection_counts = {}
            total_detections = 0
            
            try:
                if hasattr(results, 'xyxy') and len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                    # YOLOv5 style results
                    for result in results.xyxy[0]:
                        cls_id = int(result[5])
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1
                elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                    # YOLOv8 style results
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1
                
                # Log detection count
                if total_detections > 0:
                    # 检测到目标时，输出目标总数
                    self.log_update.emit(f"Обнаружено {total_detections} объект(ов)")
                    
                    # 逐类输出各目标的检测数量
                    for cls_name, count in detection_counts.items():
                        self.log_update.emit(f"  - {cls_name}: {count} шт.")
                else:
                    # 未检测到任何目标
                    self.log_update.emit("Не обнаружено ни одного объекта")
            except Exception as e:
                # 统计检测结果时发生错误
                self.log_update.emit(f"Ошибка при подсчёте результатов обнаружения: {str(e)}")
            
            # Save result image
            result_path = os.path.join(results_dir, f"result_{os.path.basename(img_path)}")
            
            if self.save_results:
                try:
                    result_saved = False
                    
                    # First try with Ultralytics v8 API
                    if hasattr(results[0], 'plot'):
                        # YOLOv8 或 yolo12 的 API
                        self.log_update.emit("Используется API YOLOv8/yolo12 для отрисовки результатов обнаружения")
                        result_img = results[0].plot()
                        cv2.imwrite(result_path, result_img)
                        self.log_update.emit(f"Результат сохранён в: {result_path}")
                        result_saved = True

                    # 然后尝试使用 Ultralytics v5 的 API
                    elif hasattr(results, 'render'):
                        # YOLOv5 的 API
                        self.log_update.emit("Используется API YOLOv5 для отрисовки результатов обнаружения")
                        results.render()  # 更新 results.imgs，在图像上绘制边框
                        if hasattr(results, 'imgs') and len(results.imgs) > 0:
                            cv2.imwrite(result_path, results.imgs[0])
                            self.log_update.emit(f"Результат сохранён в: {result_path}")
                            result_saved = True

                    # 如果前面都无法保存，则手动绘制作为兜底方案
                    if not result_saved:
                        self.log_update.emit("Используется OpenCV для ручной отрисовки результатов обнаружения")
                        
                        # 读取原始图像
                        img = cv2.imread(img_path)
                        if img is None:
                            self.log_update.emit(f"Не удалось прочитать изображение: {img_path}")
                            return
                            
                        # Draw results on the image (for YOLOv5 format)
                        if hasattr(results, 'xyxy') and len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                            for result in results.xyxy[0]:
                                # result format: (x1, y1, x2, y2, confidence, class)
                                x1, y1, x2, y2, conf, cls = result.tolist()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Draw bounding box
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Add label
                                cls_name = self.model.names[int(cls)] if hasattr(self.model, 'names') else f"Class {int(cls)}"
                                label = f"{cls_name} {conf:.2f}"
                                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Draw results on the image (for YOLOv8 format)
                        elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                            for box in results[0].boxes:
                                # Get coordinates
                                xyxy = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = map(int, xyxy)
                                
                                # Get confidence and class
                                conf = float(box.conf[0])
                                cls_id = int(box.cls[0])
                                cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                                
                                # Draw bounding box
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Add label
                                label = f"{cls_name} {conf:.2f}"
                                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Save the image
                        cv2.imwrite(result_path, img)
                        self.log_update.emit(f"Результат сохранён в: {result_path}")
                except Exception as e:
                    self.log_update.emit(f"Ошибка при сохранении результирующего изображения: {str(e)}\n{traceback.format_exc()}")
            
            # Update display with result image
            if os.path.exists(result_path):
                self.image_update.emit(result_path)
            else:
                self.image_update.emit(img_path)  # Fallback to input image
            
            # Update status with detection summary
            stats_text = f"Обнаружено {total_detections} объектов\n"
            if detection_counts:
                for cls_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
                    stats_text += f"{cls_name}: {count} 个\n"
            self.stats_update.emit(stats_text)
            
            # Update progress to 100%
            self.progress_update.emit(100)
            
        except Exception as e:
            self.log_update.emit(f"Произошла ошибка в процессе вывода: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def run_folder_inference(self):
        """Run inference on all images in a folder."""
        self.log_update.emit("Выполнить инференс для изображений в папке...")
        
        # Find all images in the folder
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for root, _, files in os.walk(self.input_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            raise ValueError("В папке не найдено файлов изображений")
        
        self.log_update.emit(f"Найдено {len(image_files)} изображений")
        
        # Create output directory
        results_dir = os.path.join(self.output_dir, 'inference_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize detection counts
        detection_counts = {}
        total_detections = 0
        
        # Process images
        for i, img_path in enumerate(image_files):
            if self.should_stop:
                self.log_update.emit("Распознавание остановлено")
                return
            
            # Update progress
            progress = int((i + 1) / len(image_files) * 100)
            self.progress_update.emit(progress)
            
            # Log current file
            self.log_update.emit(f"处理 {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
            
            # Run inference
            try:
                # Check if this is a yolo12 model
                is_yolo12 = 'yolo12' in os.path.basename(self.model_path).lower()
                
                # Try to use the appropriate API
                if is_yolo12:
                    # yolo12 API requires specific parameters
                    results = self.model(img_path, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                else:
                    # First try with newer Ultralytics v8 API
                    try:
                        results = self.model(img_path, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                    except TypeError as e:
                        # Log error and try with compatible parameters
                        self.log_update.emit(f"Ошибка параметров API: {str(e)}, попытка использовать параметры с обратной совместимостью")
                        # Use older parameter name (size)
                        results = self.model(img_path, size=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
            except Exception as e:
                self.log_update.emit(f"Не удалось выполнить инференс для изображения {os.path.basename(img_path)}: {str(e)}")
                continue
            
            # Count detections
            try:
                if hasattr(results, 'xyxy') and len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                    # YOLOv5 style results
                    for result in results.xyxy[0]:
                        cls_id = int(result[5])
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1
                elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                    # YOLOv8 style results
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1
            except Exception as e:
                self.log_update.emit(f"Ошибка при подсчёте результатов проверки: {str(e)}")
            
            # Save results
            if self.save_results:
                try:
                    result_path = os.path.join(results_dir, f"result_{os.path.basename(img_path)}")
                    result_saved = False
                    
                    # First try with Ultralytics v8 API
                    if hasattr(results[0], 'plot'):
                        # YOLOv8 API
                        result_img = results[0].plot()
                        cv2.imwrite(result_path, result_img)
                        result_saved = True
                    # Then try with Ultralytics v5 API
                    elif hasattr(results, 'render'):
                        # YOLOv5 API
                        results.render()  # Updates results.imgs with boxes
                        if hasattr(results, 'imgs') and len(results.imgs) > 0:
                            cv2.imwrite(result_path, results.imgs[0])
                            result_saved = True
                    
                    # Manual save as fallback
                    if not result_saved:
                        # Load original image
                        img = cv2.imread(img_path)
                        if img is None:
                            self.log_update.emit(f"Не удалось прочитать изображение: {img_path}")
                            continue
                            
                        # Draw results on the image (for YOLOv5 format)
                        if hasattr(results, 'xyxy') and len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                            for result in results.xyxy[0]:
                                # result format: (x1, y1, x2, y2, confidence, class)
                                x1, y1, x2, y2, conf, cls = result.tolist()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Draw bounding box
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Add label
                                cls_name = self.model.names[int(cls)] if hasattr(self.model, 'names') else f"Class {int(cls)}"
                                label = f"{cls_name} {conf:.2f}"
                                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Draw results on the image (for YOLOv8 format)
                        elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                            for box in results[0].boxes:
                                # Get coordinates
                                xyxy = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = map(int, xyxy)
                                
                                # Get confidence and class
                                conf = float(box.conf[0])
                                cls_id = int(box.cls[0])
                                cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                                
                                # Draw bounding box
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Add label
                                label = f"{cls_name} {conf:.2f}"
                                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Save the image
                        cv2.imwrite(result_path, img)
                except Exception as e:
                    self.log_update.emit(f"Ошибка при сохранении результирующего изображения: {str(e)}")
            
            # Update display with current result image (every 10% or last image)
            if i % max(1, len(image_files) // 10) == 0 or i == len(image_files) - 1:
                if self.save_results:
                    result_path = os.path.join(results_dir, f"result_{os.path.basename(img_path)}")
                    if os.path.exists(result_path):
                        self.image_update.emit(result_path)
                    else:
                        self.image_update.emit(img_path)  # Fallback to input image
                else:
                    self.image_update.emit(img_path)
        
        # Update statistics with summary
        stats_text = f"Всего изображений: {len(image_files)}\nВсего обнаруженных объектов: {total_detections}\n\n"
        if detection_counts:
            for cls_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                stats_text += f"{cls_name}: {count}个 ({percentage:.1f}%)\n"
        
        self.stats_update.emit(stats_text)
        
        # Log summary
        self.log_update.emit(f"Инференция завершена! Обработано {len(image_files)} изображений, обнаружено {total_detections} объектов")
        self.log_update.emit(f"Результаты сохранены в: {results_dir}")
        
        if detection_counts:
            self.log_update.emit("Распределение категорий:")
            for cls_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                self.log_update.emit(f"  {cls_name}: {count}个 ({percentage:.1f}%)")
    
    def run_multiple_images_inference(self):
        """Run inference on multiple selected images."""
        self.log_update.emit("Выполнение вывода по нескольким изображениям...")
        
        # Split input paths
        image_paths = self.input_path.split(';')
        self.log_update.emit(f"Всего {len(image_paths)} изображений нужно обработать")
        
        # Set up output directory
        results_dir = os.path.join(self.output_dir, 'inference_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize detection counts
        detection_counts = {}
        total_detections = 0
        
        # Process images
        for i, img_path in enumerate(image_paths):
            if self.should_stop:
                self.log_update.emit("Рассуждение остановлено")
                return
            
            # Update progress
            progress = int((i + 1) / len(image_paths) * 100)
            self.progress_update.emit(progress)
            
            # Log current file
            self.log_update.emit(f"Обработка {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            # Run inference
            try:
                # Check if this is a yolo12 model
                is_yolo12 = 'yolo12' in os.path.basename(self.model_path).lower()
                
                # Try to use the appropriate API
                if is_yolo12:
                    # yolo12 API requires specific parameters
                    results = self.model(img_path, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                else:
                    # First try with newer Ultralytics v8 API
                    try:
                        results = self.model(img_path, imgsz=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
                    except TypeError as e:
                        # Log error and try with compatible parameters
                        self.log_update.emit(f"Ошибка параметров API: {str(e)}, попытка использовать обратно совместимые параметры")
                        # Use older parameter name (size)
                        results = self.model(img_path, size=self.img_size, conf=self.conf_thresh, iou=self.iou_thresh)
            except Exception as e:
                self.log_update.emit(f"Не удалось выполнить инференцию для изображения {os.path.basename(img_path)}: {str(e)}")
                continue
            
            # Count detections
            try:
                if hasattr(results, 'xyxy') and len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                    # YOLOv5 style results
                    for result in results.xyxy[0]:
                        cls_id = int(result[5])
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1
                elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                    # YOLOv8 style results
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                        total_detections += 1
            except Exception as e:
                self.log_update.emit(f"Ошибка при подсчёте результатов проверки: {str(e)}")
            
            # Save results
            if self.save_results:
                try:
                    result_path = os.path.join(results_dir, f"result_{os.path.basename(img_path)}")
                    result_saved = False
                    
                    # First try with Ultralytics v8 API
                    if hasattr(results[0], 'plot'):
                        # YOLOv8 API
                        result_img = results[0].plot()
                        cv2.imwrite(result_path, result_img)
                        result_saved = True
                    # Then try with Ultralytics v5 API
                    elif hasattr(results, 'render'):
                        # YOLOv5 API
                        results.render()  # Updates results.imgs with boxes
                        if hasattr(results, 'imgs') and len(results.imgs) > 0:
                            cv2.imwrite(result_path, results.imgs[0])
                            result_saved = True
                    
                    # Manual save as fallback
                    if not result_saved:
                        # Load original image
                        img = cv2.imread(img_path)
                        if img is None:
                            self.log_update.emit(f"Не удалось прочитать изображение: {img_path}")
                            continue
                            
                        # Draw results on the image (for YOLOv5 format)
                        if hasattr(results, 'xyxy') and len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                            for result in results.xyxy[0]:
                                # result format: (x1, y1, x2, y2, confidence, class)
                                x1, y1, x2, y2, conf, cls = result.tolist()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Draw bounding box
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Add label
                                cls_name = self.model.names[int(cls)] if hasattr(self.model, 'names') else f"Class {int(cls)}"
                                label = f"{cls_name} {conf:.2f}"
                                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Draw results on the image (for YOLOv8 format)
                        elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                            for box in results[0].boxes:
                                # Get coordinates
                                xyxy = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = map(int, xyxy)
                                
                                # Get confidence and class
                                conf = float(box.conf[0])
                                cls_id = int(box.cls[0])
                                cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else f"Class {cls_id}"
                                
                                # Draw bounding box
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Add label
                                label = f"{cls_name} {conf:.2f}"
                                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Save the image
                        cv2.imwrite(result_path, img)
                except Exception as e:
                    self.log_update.emit(f"Ошибка при сохранении результирующего изображения: {str(e)}\n{traceback.format_exc()}")
            
            # Update display with current result image (every 10% or last image)
            if i % max(1, len(image_paths) // 10) == 0 or i == len(image_paths) - 1:
                if self.save_results:
                    result_path = os.path.join(results_dir, f"result_{os.path.basename(img_path)}")
                    if os.path.exists(result_path):
                        self.image_update.emit(result_path)
                    else:
                        self.image_update.emit(img_path)  # Fallback to input image
                else:
                    self.image_update.emit(img_path)
        
        # Update statistics with summary
        stats_text = f"Всего изображений: {len(image_paths)}\nВсего обнаруженных объектов: {total_detections}\n\n"
        if detection_counts:
            for cls_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                stats_text += f"{cls_name}: {count}个 ({percentage:.1f}%)\n"
        
        self.stats_update.emit(stats_text)
        
        # Log summary
        self.log_update.emit(f"Инференция завершена! Обработано {len(image_paths)} изображений, обнаружено {total_detections} объектов")
        self.log_update.emit(f"Результаты сохранены в: {results_dir}")
        
        if detection_counts:
            self.log_update.emit("Распределение категорий:")
            for cls_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                self.log_update.emit(f"  {cls_name}: {count} штуки ({percentage:.1f}%)") 