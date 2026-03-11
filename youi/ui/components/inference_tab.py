import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QFileDialog, QComboBox, QLineEdit, 
                            QSpinBox, QDoubleSpinBox, QGroupBox, QCheckBox, 
                            QMessageBox, QProgressBar, QTextEdit, QRadioButton,
                            QButtonGroup, QFormLayout, QGridLayout, QSplitter,
                            QScrollArea, QTabWidget, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QCoreApplication, QTimer
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QApplication

from utils.inference_worker import InferenceWorker
from utils.theme_manager import ThemeManager

from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl




class InferenceTab(QWidget):
    """Tab for YOLO model inference on images or folders."""
    
    def __init__(self):
        super().__init__()
        self.is_inferencing = False
        self.inference_worker = None
        self.inference_thread = None
        
        # Default paths (will be updated from settings if available)
        self.default_output_dir = ""
        self.default_model_path = ""
        
        # Current preview image path
        self.current_preview_image = None
        self.original_pixmap = None
        
        # Set up UI
        self.setup_ui()
        self.apply_theme_styles()  # 初始化时应用主题样式
        
        # Install event filter to handle resize events
        self.installEventFilter(self)
        
    def setup_ui(self):
        """Create and arrange UI elements."""
        main_layout = QVBoxLayout(self)
        
        # Create a splitter with left panel for settings and right panel for results
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel - Settings
        settings_panel = QWidget()
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(10, 10, 5, 10)  # Reduce horizontal margins
        
        # Model section
        model_group = QGroupBox("Настройки модели")   # 模型设置
        model_layout = QFormLayout()

        # Model selection
        self.model_path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_btn = QPushButton("Обзор...")    # 浏览
        self.model_path_layout.addWidget(self.model_path_edit)
        self.model_path_layout.addWidget(self.model_path_btn)

        # Confidence threshold
        self.conf_thresh_spin = QDoubleSpinBox()
        self.conf_thresh_spin.setRange(0.1, 1.0)
        self.conf_thresh_spin.setValue(0.25)
        self.conf_thresh_spin.setSingleStep(0.05)
        
        # IoU threshold
        self.iou_thresh_spin = QDoubleSpinBox()
        self.iou_thresh_spin.setRange(0.1, 1.0)
        self.iou_thresh_spin.setValue(0.45)
        self.iou_thresh_spin.setSingleStep(0.05)
        
        # Image size
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 1280)
        self.img_size_spin.setValue(640)
        self.img_size_spin.setSingleStep(32)
        
        # Add widgets to form layout
        model_layout.addRow("Путь к модели:", self.model_path_layout)  # 模型路径:
        model_layout.addRow("Порог уверенности:", self.conf_thresh_spin)  # 置信度阈值:
        model_layout.addRow("Порог IoU:", self.iou_thresh_spin)          # IoU阈值:
        model_layout.addRow("Размер изображения:", self.img_size_spin)  # 图像尺寸:
        model_group.setLayout(model_layout)

        # Inference mode section
        mode_group = QGroupBox("Режим распознавания")  # 推理模式
        mode_layout = QFormLayout()

        # Mode selection
        self.mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup(self)

        self.image_radio = QRadioButton("Изображения")  # 图片模式
        self.folder_radio = QRadioButton("Папки")  # 文件夹模式
        self.video_radio = QRadioButton("Видео")  # 视频模式
        self.camera_radio = QRadioButton("Камеры")  # 摄像头模式

        self.mode_group.addButton(self.image_radio)
        self.mode_group.addButton(self.folder_radio)
        self.mode_group.addButton(self.video_radio)
        self.mode_group.addButton(self.camera_radio)

        self.mode_layout.addWidget(self.image_radio)
        self.mode_layout.addWidget(self.folder_radio)
        self.mode_layout.addWidget(self.video_radio)
        self.mode_layout.addWidget(self.camera_radio)

        self.image_radio.setChecked(True)

        # Input selection
        self.input_layout = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True) 
        self.input_btn = QPushButton("Обзор...")   # 浏览...
        self.input_layout.addWidget(self.input_edit)
        self.input_layout.addWidget(self.input_btn)

        # Output directory
        self.output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_btn = QPushButton("Обзор...")   # 浏览...
        self.output_dir_layout.addWidget(self.output_dir_edit)
        self.output_dir_layout.addWidget(self.output_dir_btn)

        # Add widgets to form layout
        mode_layout.addRow("Режим распознавания:", self.mode_layout)  # 推理模式:
        mode_layout.addRow("Путь ввода:", self.input_layout)      # 输入路径:
        mode_layout.addRow("Выходной каталог:", self.output_dir_layout)  # 输出目录:

        # Save results option
        self.save_results = QCheckBox("Сохранить результаты распознавания")  # 保存检测结果
        self.save_results.setChecked(True)
        mode_layout.addRow("", self.save_results)

        # Add view results button
        self.view_results_btn = QPushButton("Просмотреть результаты распознавания")  # 查看检测结果
        self.view_results_btn.setEnabled(False)  # Disabled until inference is completed
        mode_layout.addRow("", self.view_results_btn)

        mode_group.setLayout(mode_layout)

        # Control buttons
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Запустить Распознавание")  # 开始推理
        self.start_btn.setMinimumHeight(40)
        self.stop_btn = QPushButton("Остановить Распознавание")  # 停止推理
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)

        # Task info group
        info_group = QGroupBox("Информация о задаче")  # 任务信息
        info_layout = QVBoxLayout()

        # Status display
        self.status_label = QLabel("Статус: Готово")  # 状态: 就绪
        self.status_label.setStyleSheet("font-weight: bold; color: #333;")
        info_layout.addWidget(self.status_label)

        # Stats display
        self.stats_label = QLabel("Ожидание запуска Распознавания...")  # 等待推理开始...
        info_layout.addWidget(self.stats_label)

        # Progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Прогресс:"))  # 进度:
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        info_layout.addLayout(progress_layout)
        
        info_group.setLayout(info_layout)
        
        # Add sections to settings panel
        settings_layout.addWidget(model_group)
        settings_layout.addWidget(mode_group)
        settings_layout.addLayout(control_layout)
        settings_layout.addWidget(info_group)
        settings_layout.addStretch(1)  # Add stretch to push everything up
        
        # Right Panel - Results
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        
        # Right side uses vertical splitter, top for result images, bottom for terminal output
        results_splitter = QSplitter(Qt.Vertical)
        
        # Upper part: Result image area
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(5, 5, 5, 5)
        
        # Image header with title and save button
        image_header = QWidget()
        image_header_layout = QHBoxLayout(image_header)
        image_header_layout.setContentsMargins(0, 0, 0, 5)
        
        image_title = QLabel("Предпросмотр результатов распознавания")  # 检测结果预览
        image_title.setFont(QFont("", 11, QFont.Bold))
        image_header_layout.addWidget(image_title)

        self.save_img_btn = QPushButton("Сохранить")  # 保存图片
        self.save_img_btn.setMaximumWidth(120)
        self.save_img_btn.setStyleSheet("background-color: #0d1a2f; border: 1px solid #ccc; border-radius: 3px;")
        image_header_layout.addWidget(self.save_img_btn, alignment=Qt.AlignRight)
        
        image_layout.addWidget(image_header)
        
        # Image display area with overlay controls
        image_container = QWidget()
        image_container_layout = QVBoxLayout(image_container)
        image_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a custom label that handles resizing
        class ScalableImageLabel(QLabel):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.original_pixmap = None
                self.setAlignment(Qt.AlignCenter)
                self.setMinimumHeight(450)
                self.setStyleSheet("background-color: #0d1a2f; border: 1px solid #ddd; border-radius: 4px;")
                self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                
            def setPixmap(self, pixmap):
                self.original_pixmap = pixmap
                self.updatePixmap()
                
            def updatePixmap(self):
                if self.original_pixmap and not self.original_pixmap.isNull():
                    scaled_pixmap = self.original_pixmap.scaled(
                        self.width(), self.height(),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    super().setPixmap(scaled_pixmap)
                    
            def resizeEvent(self, event):
                super().resizeEvent(event)
                self.updatePixmap()
        
        # Use the custom label for image display
        self.image_display = ScalableImageLabel()

        # 设置默认图像（首次显示）
        default_image_path = os.path.join(os.path.dirname(__file__), '../assets/moren1.png')
        if os.path.exists(default_image_path):
            default_pixmap = QPixmap(default_image_path)
            self.image_display.setPixmap(default_pixmap)
            self.current_preview_image = default_image_path
        else:
            self.image_display.setText("Ожидание распознавания...")  # 等待检测...


        # 用于播放视频的控件（隐藏，视频推理完成后显示）
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(450)
        self.video_widget.setStyleSheet("background-color: black; border: 1px solid #444; border-radius: 4px;")
        self.video_widget.setVisible(False)

        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)

        # 添加到 image_container_layout
        image_container_layout.addWidget(self.video_widget)

        # Put image directly in the layout without scroll area
        image_container_layout.addWidget(self.image_display, 1)
        
        # Image browser controls - moved to image area and improved visibility
        self.img_browser_controls = QWidget()
        self.img_browser_controls.setStyleSheet("""
            QWidget {
                background-color: rgba(240, 240, 240, 0.85);
                border-top: 1px solid #ccc;
            }
            QPushButton {
                padding: 5px 10px;
                background-color: #121a22;
                border: 1px solid #ccc;
                border-radius: 4px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #121a22;
            }
            QLabel {
                font-weight: bold;
                color: #121a22;
            }
        """)
        img_browser_layout = QHBoxLayout(self.img_browser_controls)
        img_browser_layout.setContentsMargins(10, 10, 10, 10)
        
        self.prev_img_btn = QPushButton("« Предыдущее изображение")  # « 上一张
        self.prev_img_btn.setCursor(Qt.PointingHandCursor)

        self.next_img_btn = QPushButton("Следующее изображение »")  # 下一张 »
        self.next_img_btn.setCursor(Qt.PointingHandCursor)

        self.img_counter_label = QLabel("0/0")
        self.img_counter_label.setAlignment(Qt.AlignCenter)

        self.close_browser_btn = QPushButton("Закрыть браузер")  # 关闭浏览器
        self.close_browser_btn.setCursor(Qt.PointingHandCursor)
        
        img_browser_layout.addWidget(self.prev_img_btn)
        img_browser_layout.addWidget(self.img_counter_label)
        img_browser_layout.addWidget(self.next_img_btn)
        img_browser_layout.addWidget(self.close_browser_btn)
        
        self.img_browser_controls.setVisible(False)
        image_container_layout.addWidget(self.img_browser_controls)
        
        # Add the image container to the image layout
        image_layout.addWidget(image_container)
        
        # Lower part: Terminal output
        terminal_widget = QWidget()
        terminal_layout = QVBoxLayout(terminal_widget)
        terminal_layout.setContentsMargins(5, 5, 5, 5)
        
        # Terminal header
        terminal_header = QWidget()
        terminal_header_layout = QHBoxLayout(terminal_header)
        terminal_header_layout.setContentsMargins(0, 0, 0, 5)
        
        terminal_title = QLabel("Вывод терминала")  # 终端输出
        terminal_title.setFont(QFont("", 11, QFont.Bold))
        terminal_header_layout.addWidget(terminal_title)

        clear_btn = QPushButton("Очистить")  # 清除
        clear_btn.setMaximumWidth(100)
        clear_btn.setStyleSheet("background-color: #0d1a2f; border: 1px solid #ccc; border-radius: 3px;")
        clear_btn.clicked.connect(self.clear_terminal)
        terminal_header_layout.addWidget(clear_btn, alignment=Qt.AlignRight)
        
        terminal_layout.addWidget(terminal_header)
        
        # Terminal text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        self.log_text.document().setDefaultStyleSheet("pre {margin: 0; font-family: monospace;}")
        self.log_text.setStyleSheet("background-color: #0d1a2f; border: 1px solid #ddd; border-radius: 4px; font-family: Consolas, monospace;")
        terminal_layout.addWidget(self.log_text)
        
        # Add components to results splitter
        results_splitter.addWidget(image_widget)
        results_splitter.addWidget(terminal_widget)
        results_splitter.setSizes([600, 200])  # Adjust size ratio to favor image area
        
        # Set stretch factors for the results splitter
        results_splitter.setStretchFactor(0, 3)  # Image area gets more stretch
        results_splitter.setStretchFactor(1, 1)  # Terminal area gets less stretch
        
        # Add splitter to results panel
        results_layout.addWidget(results_splitter)
        
        # Add panels to main splitter
        main_splitter.addWidget(settings_panel)
        main_splitter.addWidget(results_panel)
        
        # Add main splitter to layout
        main_layout.addWidget(main_splitter)
        
        # Set initial size proportions - make results panel larger
        main_splitter.setSizes([200, 800])  # Adjust size ratio to give more space to results
        
        # Set stretch factors to maintain proportions during resize
        main_splitter.setStretchFactor(0, 0)  # Settings panel doesn't stretch
        main_splitter.setStretchFactor(1, 1)  # Results panel gets all the stretch
        
        # Connect signals
        self.connect_signals()

    def play_result_video(self, video_path):
        """播放推理结果视频"""
        if not os.path.exists(video_path):
            self.log_message(f"Видеофайл не существует: {video_path}")  # 视频文件不存在: {video_path}
            return



        self.image_display.setVisible(False)
        self.video_widget.setVisible(True)

        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.media_player.play()

        self.status_label.setText("Статус: Воспроизведение видео")  # 状态: 正在播放视频
        self.status_label.setStyleSheet("font-weight: bold; color: #009688;")
        self.log_message(f"Воспроизведение видео распознавания: {video_path}")  # 播放推理视频: {video_path}

    def connect_signals(self):
        """Connect UI signals to their respective slots."""
        # Mode selection
        self.image_radio.toggled.connect(self.update_mode)
        self.folder_radio.toggled.connect(self.update_mode)
        
        # Path selection
        self.model_path_btn.clicked.connect(self.select_model_path)
        self.input_btn.clicked.connect(self.select_input_path)
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        
        # Control buttons
        self.start_btn.clicked.connect(self.start_inference)
        self.stop_btn.clicked.connect(self.stop_inference)
        self.save_img_btn.clicked.connect(self.save_preview_image)
        
        # Image browser controls
        self.view_results_btn.clicked.connect(self.open_image_browser)
        self.prev_img_btn.clicked.connect(self.show_prev_image)
        self.next_img_btn.clicked.connect(self.show_next_image)
        self.close_browser_btn.clicked.connect(self.close_image_browser)
        
        # 监听主题切换信号
        from ui.main_window import MainWindow
        main_window = self.parentWidget()
        while main_window and not isinstance(main_window, MainWindow):
            main_window = main_window.parentWidget()
        if main_window and hasattr(main_window, 'settings_tab'):
            main_window.settings_tab.theme_changed.connect(lambda _: self.apply_theme_styles())

    def update_mode(self, checked):
        if self.image_radio.isChecked():
            self.input_btn.setText("Выбрать изображение...")  # 选择图片...
            self.input_edit.setPlaceholderText("Выберите изображение для детекции")  # 选择要检测的图片
        elif self.folder_radio.isChecked():
            self.input_btn.setText("Выбрать папку...")  # 选择文件夹...
            self.input_edit.setPlaceholderText("Выберите папку с изображениями")  # 选择包含图片的文件夹
        elif self.video_radio.isChecked():
            self.input_btn.setText("Выбрать видеофайл...")  # 选择视频文件...
            self.input_edit.setPlaceholderText("Выберите видео для детекции")  # 选择要检测的视频
        elif self.camera_radio.isChecked():
            self.input_btn.setText("Открыть камеру")  # 打开摄像头
            self.input_edit.setPlaceholderText("Номер камеры, например 0")  # 摄像头编号，如 0
        self.input_edit.clear()

    def select_model_path(self):
        """Открыть диалог выбора файла модели."""  # 打开对话框以选择模型文件
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать файл модели",  # 选择模型文件
            "",
            "Файлы моделей (*.pt *.pth *.weights);;Все файлы (*)"  # 模型文件 (*.pt *.pth *.weights);;所有文件 (*)
        )
        if file_path:
            self.model_path_edit.setText(file_path)

    def select_input_path(self):
        if self.image_radio.isChecked():
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Выбрать изображения",  # 选择图片
                "",
                "Файлы изображений (*.jpg *.jpeg *.png *.bmp);;Все файлы (*)"  # 图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)
            )
            if file_paths:
                self.input_edit.setText(";".join(file_paths))
        elif self.folder_radio.isChecked():
            dir_path = QFileDialog.getExistingDirectory(
                self,
                "Выбрать папку с изображениями"  # 选择包含图片的文件夹
            )
            if dir_path:
                self.input_edit.setText(dir_path)
        elif self.video_radio.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Выбрать видеофайл",  # 选择视频文件
                "",
                "Видеофайлы (*.mp4 *.avi *.mov);;Все файлы (*)"  # 视频文件 (*.mp4 *.avi *.mov);;所有文件 (*)
            )
            if file_path:
                self.input_edit.setText(file_path)
        elif self.camera_radio.isChecked():
            self.input_edit.setText("0")  # 默认摄像头编号为 0

    def select_output_dir(self):
        """Open dialog to select output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Выбрать выходной каталог")  # 选择输出目录
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def start_inference(self):
        """Validate inputs and start inference in a separate thread."""
        # Validate inputs
        if not self.validate_inputs():
            return
        
        # Disable UI elements during inference
        self.set_ui_enabled(False)
        self.is_inferencing = True
        
        # Reset status and progress
        self.status_label.setText("Статус: Подготовка")  # 状态: 准备中
        self.status_label.setStyleSheet("font-weight: bold; color: #333;")
        self.progress_bar.setValue(0)
        self.stats_label.setText("Загрузка модели...")  # 正在加载模型...

        # Clear displays  # 清空显示区域
        self.log_text.clear()
        self.log_message("Подготовка задачи распознавания...")  # 准备推理任务...
        
        # Create worker and thread
        self.inference_worker = InferenceWorker(
            model_path=self.model_path_edit.text(),
            input_path=self.input_edit.text(),
            output_dir=self.output_dir_edit.text(),
            is_folder_mode=self.folder_radio.isChecked(),
            is_video_mode=self.video_radio.isChecked(),
            is_camera_mode=self.camera_radio.isChecked(),
            conf_thresh=self.conf_thresh_spin.value(),
            iou_thresh=self.iou_thresh_spin.value(),
            img_size=self.img_size_spin.value(),
            save_results=self.save_results.isChecked()
        )

        self.inference_thread = QThread()
        self.inference_worker.moveToThread(self.inference_thread)
        
        # Connect signals
        self.inference_worker.progress_update.connect(self.update_progress)
        self.inference_worker.log_update.connect(self.log_message)
        self.inference_worker.stats_update.connect(self.update_stats)
        self.inference_worker.stats_update.connect(self.log_message)
        self.inference_worker.image_update.connect(self.update_image)
        self.inference_worker.inference_complete.connect(self.on_inference_complete)
        self.inference_worker.inference_error.connect(self.on_inference_error)
        self.inference_thread.started.connect(self.inference_worker.run)
        
        # Start inference
        self.inference_thread.start()
    
    def stop_inference(self):
        """Остановить процесс инференса."""  # 停止推理过程
        self.log_message("Остановка распознавания (пожалуйста, подождите)...")  # 正在停止推理(请稍候)...

        # Update status  # 更新状态
        self.status_label.setText("Статус: Остановка...")  # 状态: 正在停止...
        self.status_label.setStyleSheet("font-weight: bold; color: #f57900;")

        if self.inference_worker:
            self.inference_worker.stop()

    def on_inference_complete(self):
        self.log_message("распознавание завершён")  # 推理完成
        self.is_inferencing = False
        self.set_ui_enabled(True)
        self.status_label.setText("Статус: Завершено")  # 状态: 已完成
        self.progress_bar.setValue(100)

        # Не использовать просмотрщик изображений для отображения результатов видео или камеры
        # 不使用图片浏览器显示视频或摄像头结果
        if self.video_radio.isChecked() or self.camera_radio.isChecked():
            output_video_path = os.path.join(
                self.output_dir_edit.text(),
                'inference_results',
                'video_output.avi' if self.video_radio.isChecked() else 'camera_output.avi'
            )
            if os.path.exists(output_video_path):
                self.play_result_video(output_video_path)
            else:
                self.log_message("Выходное видео не найдено, воспроизведение невозможно.")  # 未找到输出视频，无法播放。
            return

        if self.save_results.isChecked():
            self.view_results_btn.setEnabled(True)
            results_dir = os.path.join(self.output_dir_edit.text(), 'inference_results')
            if os.path.exists(results_dir):
                image_files = [
                    f for f in os.listdir(results_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
                ]
                if image_files and len(image_files) > 0:
                    reply = QMessageBox.question(
                        self,
                        'Просмотр результатов',  # 查看结果
                        f"Распознавания завершён. Хотите сразу просмотреть результаты детекции?\n(Всего {len(image_files)} изображений)", 
                          # 推理已完成，是否立即查看检测结果？\n(共 {len(image_files)} 张图片)
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    if reply == QMessageBox.Yes:
                        self.open_image_browser()

        # Clean up thread  # 清理线程
        self.clean_up_thread()

    def on_inference_error(self, error_msg):
        """Обработать ошибку инференса."""  # 处理推理错误
        self.is_inferencing = False
        self.clean_up_thread()
        self.set_ui_enabled(True)

        # Update status  # 更新状态
        self.status_label.setText("Статус: Ошибка")  # 状态: 出错
        self.status_label.setStyleSheet("font-weight: bold; color: red;")

        self.log_message(f"Ошибка: {error_msg}")  # 错误: {error_msg}
        QMessageBox.critical(
            self,
            "Ошибка распознавания",  # 推理错误
            f"Во время распознавания произошла ошибка:\n{error_msg}"  # 推理过程中发生错误:\n{error_msg}
        )
    
    def clean_up_thread(self):
        """Clean up thread and worker resources."""
        if self.inference_thread:
            self.inference_thread.quit()
            self.inference_thread.wait()
            self.inference_thread = None
            self.inference_worker = None
    
    def update_progress(self, progress):
        """Обновить индикатор прогресса."""  # 更新进度条
        # Update progress bar  # 更新进度条
        self.progress_bar.setValue(progress)

        # Update status label  # 更新状态标签
        self.status_label.setText(f"Статус: Выполняется распознавание ({progress}%)")  # 状态: 推理中 ({progress}%)

        # Log progress at 10% intervals  # 每 10% 记录一次进度日志
        if progress % 10 == 0:
            self.log_message(f"Прогресс распознавания: {progress}%")  # 推理进度: {progress}%

    def log_message(self, message):
        """Добавить сообщение в журнал."""  # 添加消息到日志显示
        self.log_text.append(message)
        # Also print to stdout for terminal redirection  # 同时输出到 stdout 以便终端重定向
        print(f"[распознавание] {message}")  # [推理] {message}

        # Auto-scroll to bottom  # 自动滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def update_stats(self, stats_text):
        """Update statistics display."""
        self.stats_label.setText(stats_text)
    
    def update_image(self, image_path):
        """Update image display with detection result."""
        if os.path.exists(image_path):
            # Save current image path
            self.current_preview_image = image_path
            
            # Load the image and set it to our scalable label
            pixmap = QPixmap(image_path)
            self.image_display.setPixmap(pixmap)
                
            # If this is being called from the image browser, ensure controls are visible
            if hasattr(self, 'image_files') and self.image_files:
                self.img_browser_controls.setVisible(True)
        else:
            self.log_message(f"Невозможно отобразить изображение: {image_path} (файл не существует)")  # 无法显示图像: {image_path} (文件不存在)
            self.image_display.clear()
            self.image_display.setStyleSheet("background-color: #f0f0f0;")
            self.current_preview_image = None
    
    def resizeEvent(self, event):
        """Main widget resize event - pass to parent"""
        super().resizeEvent(event)
        
    def set_ui_enabled(self, enabled):
        """Enable or disable UI elements during inference."""
        # Model settings
        self.model_path_btn.setEnabled(enabled)
        self.conf_thresh_spin.setEnabled(enabled)
        self.iou_thresh_spin.setEnabled(enabled)
        self.img_size_spin.setEnabled(enabled)
        
        # Mode settings
        self.image_radio.setEnabled(enabled)
        self.folder_radio.setEnabled(enabled)
        self.input_btn.setEnabled(enabled)
        self.output_dir_btn.setEnabled(enabled)
        self.save_results.setEnabled(enabled)
        
        # Start/stop buttons
        self.start_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)
        
        # Don't disable view results button when starting/stopping inference
        # It should remain enabled if results are available
        
        # Update progress bar status
        if not enabled:
            self.progress_bar.setStyleSheet("QProgressBar { text-align: center; }")
        else:
            # Reset status  # 重置状态
            self.status_label.setText("Статус: Готово")  # 状态: 就绪
            self.status_label.setStyleSheet("font-weight: bold; color: #333;")
            self.progress_bar.setValue(0)
            self.stats_label.setText("Ожидание начала распознавания...")  # 等待推理开始...



    def validate_inputs(self):

        if self.video_radio.isChecked():
            video_path = self.input_edit.text()
            if not os.path.isfile(video_path):
                QMessageBox.warning(self, "Недопустимый ввод", "Выбранный видеофайл недействителен.")  # 无效输入 / 所选视频文件无效。
                return False
            if not video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                QMessageBox.warning(self, "Ошибка формата", "Неподдерживаемый формат видео.")  # 格式错误 / 不支持的视频格式。
                return False

        if self.camera_radio.isChecked():
            try:
                cam_index = int(self.input_edit.text())
                if cam_index < 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Недопустимый ввод", "Введите корректный номер камеры (например, 0).")  # 无效输入 / 请输入有效的摄像头编号（例如 0）。
                return False

        """Проверить пользовательские входные данные перед запуском инференса."""  # 启动推理前验证用户输入
        # Check if model path is set  # 检查是否已设置模型路径
        if not self.model_path_edit.text():
            QMessageBox.warning(self, "Отсутствует ввод", "Пожалуйста, выберите файл модели.")  # 缺少输入 / 请选择模型文件。
            return False

        # Check if input path is set  # 检查是否已设置输入路径
        if not self.input_edit.text():
            if self.image_radio.isChecked():
                QMessageBox.warning(self, "Отсутствует ввод", "Пожалуйста, выберите изображение для детекции.")  # 缺少输入 / 请选择要检测的图片。
            else:
                QMessageBox.warning(self, "Отсутствует ввод", "Пожалуйста, выберите папку с изображениями.")  # 缺少输入 / 请选择包含图片的文件夹。
            return False

        # Check if output directory is set  # 检查是否已设置输出目录
        if not self.output_dir_edit.text():
            QMessageBox.warning(self, "Отсутствует ввод", "Пожалуйста, выберите выходную папку.")  # 缺少输入 / 请选择输出目录。
            return False

        # In folder mode, check if directory exists and contains images  # 在文件夹模式下，检查目录是否存在且包含图片
        if self.folder_radio.isChecked():
            input_dir = self.input_edit.text()
            if not os.path.isdir(input_dir):
                QMessageBox.warning(self, "Недопустимый ввод", "Выбранный путь не является допустимой папкой.")  # 无效输入 / 所选路径不是有效的文件夹。
                return False
            
            # Check if directory contains any images
            has_images = False
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        has_images = True
                        break
                if has_images:
                    break
            
            if not has_images:
                QMessageBox.warning(self, "Недопустимый ввод", "Выбранная папка не содержит файлов изображений.")  # 无效输入 / 所选文件夹不包含任何图片文件。
                return False
        
        # In image mode, check if files exist and are images
        if self.image_radio.isChecked():
            image_paths = self.input_edit.text().split(';')
            invalid_paths = []
            
            for image_path in image_paths:
                # Check if file exists  # 检查文件是否存在
                if not os.path.isfile(image_path):
                    invalid_paths.append(f"{image_path} (не является допустимым файлом)")  # 不是有效的文件
                    continue

                # Check if file is an image  # 检查文件是否为图片
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                if not any(image_path.lower().endswith(ext) for ext in image_extensions):
                    invalid_paths.append(f"{image_path} (неподдерживаемый формат изображения)")  # 不是支持的图片格式

            if invalid_paths:
                error_msg = "Следующие файлы недействительны:\n" + "\n".join(invalid_paths)  # 以下文件无效:
                QMessageBox.warning(self, "Недопустимый ввод", error_msg)  # 无效输入
                return False
        
        return True
    
    def update_settings(self, settings):
        """Update tab settings based on settings from settings tab."""
        if 'default_conf_thresh' in settings:
            self.conf_thresh_spin.setValue(settings['default_conf_thresh'])
        
        if 'default_iou_thresh' in settings:
            self.iou_thresh_spin.setValue(settings['default_iou_thresh'])
        
        if 'default_img_size' in settings:
            self.img_size_spin.setValue(settings['default_img_size'])
        
        # Update default output directory
        if 'default_output_dir' in settings:
            self.default_output_dir = settings['default_output_dir']
            if self.default_output_dir and not self.output_dir_edit.text():
                self.output_dir_edit.setText(self.default_output_dir)
            
        # Update default model path
        if 'default_test_model_path' in settings:
            self.default_model_path = settings['default_test_model_path']
            if self.default_model_path and not self.model_path_edit.text():
                self.model_path_edit.setText(self.default_model_path)
    
    def clear_terminal(self):
        """Очистить вывод терминала."""  # 清空终端输出
        self.log_text.clear()

    def save_preview_image(self):
        """Сохранить изображение предпросмотра."""  # 保存预览图像
        if not hasattr(self, 'current_preview_image') or not self.current_preview_image:
            QMessageBox.information(self, "Сохранение не удалось", "Нет изображения предпросмотра для сохранения")  # 保存失败 / 没有可保存的预览图像
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение предпросмотра", "", "Файлы изображений (*.png *.jpg *.jpeg)"
        )

        if save_path:
            # Make sure file has an extension  # 确保文件有扩展名
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                save_path += '.jpg'

            import shutil
            try:
                shutil.copy2(self.current_preview_image, save_path)
                QMessageBox.information(self, "Сохранение успешно", f"Изображение сохранено в: {save_path}")  # 保存成功 / 图像已保存至:
            except Exception as e:
                QMessageBox.critical(self, "Сохранение не удалось", f"Ошибка при сохранении изображения: {str(e)}")  # 保存失败 / 保存图像时出错:

    # Add image browser functionality  # 添加图片浏览器功能
    def open_image_browser(self):
        """Открыть браузер изображений для просмотра результатов детекции."""  # 打开图片浏览器以查看检测结果
        # Get the results directory  # 获取结果目录
        results_dir = os.path.join(self.output_dir_edit.text(), 'inference_results')

        if not os.path.exists(results_dir):
            QMessageBox.warning(
                self,
                "Невозможно открыть браузер изображений",  # 无法打开图片浏览器
                f"Каталог результатов не существует: {results_dir}\nСначала запустите распознавание"  # 结果目录不存在 / 请先运行推理。
            )
            return

        # Get all image files in the directory  # 获取目录中的所有图片文件
        self.image_files = [
            f for f in os.listdir(results_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ]

        if not self.image_files:
            QMessageBox.information(
                self,
                "Нет результатов детекции",  # 没有检测结果
                "В каталоге результатов не найдено файлов изображений。"  # 结果目录中没有找到图片文件。
            )
            return
        
        # Initialize browser
        self.current_img_index = 0
        
        # Show image browser controls
        self.img_browser_controls.setVisible(True)
        
        # Update counter display
        self.img_counter_label.setText(f"1/{len(self.image_files)}")
        
        # Update status to indicate browsing mode
        self.status_label.setText("Статус: Просмотр результатов детекции")  # 状态: 浏览检测结果
        self.status_label.setStyleSheet("font-weight: bold; color: #0066cc;")

        # Log the action  # 记录操作
        self.log_message(f"Открыт браузер изображений, найдено {len(self.image_files)} изображений")  # 打开图片浏览器, 共找到 ... 张图片
                
        # Show first image
        self.display_current_image()
        
        # Make sure UI updates are processed
        QCoreApplication.processEvents()
        
    def display_current_image(self):
        """Display the current image in the browser."""
        if not hasattr(self, 'image_files') or not self.image_files:
            return
            
        # Get the results directory
        results_dir = os.path.join(self.output_dir_edit.text(), 'inference_results')
        
        # Get current image path
        img_path = os.path.join(results_dir, self.image_files[self.current_img_index])
        
        # Update counter
        self.img_counter_label.setText(f"{self.current_img_index + 1}/{len(self.image_files)}")

        # print('设置样式前:', self.img_counter_label.styleSheet())
        self.img_counter_label.setStyleSheet("color: white !important; background-color: #121a22 !important;")
        # print('设置样式后:', self.img_counter_label.styleSheet())
        self.img_counter_label.setMinimumWidth(80)
        self.img_counter_label.setMinimumHeight(40)
        # Load and display image
        self.update_image(img_path)
        
        # Update current preview image
        self.current_preview_image = img_path
        
    def show_next_image(self):
        """Show the next image in the browser."""
        if not hasattr(self, 'image_files') or not self.image_files:
            return
            
        self.current_img_index = (self.current_img_index + 1) % len(self.image_files)
        self.display_current_image()
        
    def show_prev_image(self):
        """Show the previous image in the browser."""
        if not hasattr(self, 'image_files') or not self.image_files:
            return
            
        self.current_img_index = (self.current_img_index - 1) % len(self.image_files)
        self.display_current_image()
        
    def close_image_browser(self):
        """Close the image browser."""
        self.img_browser_controls.setVisible(False)
        
        # Reset status
        self.status_label.setText("Статус: Готово")  # 状态: 就绪
        self.status_label.setStyleSheet("font-weight: bold; color: #333;")

        # Clear the displayed image  # 清除显示的图像
        self.image_display.clear()
        self.image_display.setText("Нет предпросмотра")  # 无预览
        self.current_preview_image = None 

    def eventFilter(self, obj, event):
        """Event filter - no longer needed for image scaling"""
        return super().eventFilter(obj, event) 

    def apply_theme_styles(self):
        """根据当前主题刷新所有控件样式，保证高对比度"""
        app = QApplication.instance()
        theme = app.property('theme') if app and app.property('theme') else 'tech'
        # 终端输出区
        if hasattr(self, 'log_text'):
            if theme == 'light':
                self.log_text.setStyleSheet("background-color: #f7f7f7; color: #212529; border: 1px solid #ddd; border-radius: 4px; font-family: Consolas, monospace;")
            elif theme == 'dark':
                self.log_text.setStyleSheet("background-color: #1E1E1E; color: #DCE6F0; border: 1px solid #3F3F46; border-radius: 4px; font-family: Consolas, monospace;")
            else:  # tech
                self.log_text.setStyleSheet("background-color: #121A22; color: #DCE6F0; border: 1px solid #34465A; border-radius: 4px; font-family: Consolas, monospace;")
        # 主要标签和按钮
        label_color = {'light': '#212529', 'dark': '#DCE6F0', 'tech': '#DCE6F0'}[theme]
        for label in self.findChildren(QLabel):
            label.setStyleSheet(f"color: {label_color};")
        for btn in self.findChildren(QPushButton):
            btn.setStyleSheet(btn.styleSheet() + f"color: {label_color};")
        # QGroupBox标题
        for group in self.findChildren(QGroupBox):
            group.setStyleSheet(f"QGroupBox {{ color: {label_color}; }}")
        # 进度条
        if hasattr(self, 'progress_bar'):
            if theme == 'light':
                self.progress_bar.setStyleSheet("QProgressBar { background: #f7f7f7; color: #212529; border: 1px solid #ddd; border-radius: 4px; } QProgressBar::chunk { background: #4287f5; border-radius: 3px; }")
            elif theme == 'dark':
                self.progress_bar.setStyleSheet("QProgressBar { background: #252526; color: #DCE6F0; border: 1px solid #3F3F46; border-radius: 4px; } QProgressBar::chunk { background: #007ACC; border-radius: 3px; }")
            else:
                self.progress_bar.setStyleSheet("QProgressBar { background: #151E28; color: #DCE6F0; border: 1px solid #34465A; border-radius: 5px; } QProgressBar::chunk { background: #00BFFF; border-radius: 3px; }")
        # 图像显示区
        if hasattr(self, 'image_display'):
            if theme == 'light':
                self.image_display.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd; border-radius: 4px;")
            elif theme == 'dark':
                self.image_display.setStyleSheet("background-color: #1E1E1E; border: 1px solid #3F3F46; border-radius: 4px;")
            else:
                self.image_display.setStyleSheet("background-color: #121A22; border: 1px solid #34465A; border-radius: 4px;") 