from PyQt5.QtWidgets import QWidget, QToolTip
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont

class TooltipManager:
    """管理UI元素的工具提示"""
    
    @staticmethod
    def apply_tooltips(window):
        """为主窗口的UI元素添加工具提示"""
        # 设置工具提示全局字体
        QToolTip.setFont(QFont('Segoe UI', 9))
        
        # 应用主窗口工具提示
        # TooltipManager.apply_training_tab_tooltips(window.training_tab)
        # TooltipManager.apply_testing_tab_tooltips(window.testing_tab)
        # TooltipManager.apply_inference_tab_tooltips(window.inference_tab)
        # TooltipManager.apply_settings_tab_tooltips(window.settings_tab)
    
    @staticmethod
    def apply_training_tab_tooltips(tab):
        """为训练标签页添加工具提示"""
        if not tab:
            return
            
        # Подсказки для элементов управления на вкладке обучения  # 常用训练标签页控件的工具提示
        if hasattr(tab, 'dataset_path_edit'):
            tab.dataset_path_edit.setToolTip("Укажите путь к набору данных<br><b>Горячая клавиша:</b> нет")  # 指定数据集的路径 / 快捷键: 无

        if hasattr(tab, 'model_combo'):
            tab.model_combo.setToolTip("Выберите архитектуру модели для обучения<br><b>Горячая клавиша:</b> нет")  # 选择要训练的模型架构 / 快捷键: 无

        if hasattr(tab, 'epochs_spin'):
            tab.epochs_spin.setToolTip("Количество эпох обучения<br><b>Подсказка:</b> большее число эпох обычно даёт лучший результат, но требует больше времени")  # 训练的轮数 / 提示...

        if hasattr(tab, 'batch_size_spin'):
            tab.batch_size_spin.setToolTip("Количество образцов в одном батче<br><b>Подсказка:</b> больший размер батча может ускорить обучение, но требует больше памяти GPU")  # 每批次训练的样本数量 / 提示...

        if hasattr(tab, 'start_training_btn'):
            tab.start_training_btn.setToolTip("Запустить обучение модели<br><b>Убедитесь, что все параметры указаны правильно</b>")  # 开始训练模型 / 确保所有参数设置正确

        if hasattr(tab, 'stop_training_btn'):
            tab.stop_training_btn.setToolTip("Остановить текущий процесс обучения<br><b>Предупреждение:</b> после остановки обучение нельзя будет продолжить")  # 停止当前训练进程 / 警告...

        if hasattr(tab, 'clear_log_btn'):
            tab.clear_log_btn.setToolTip("Очистить журнал обучения<br><b>Горячая клавиша:</b> Ctrl+L")  # 清除训练日志 / 快捷键: Ctrl+L
    
    @staticmethod
    def apply_testing_tab_tooltips(tab):
        """为测试标签页添加工具提示"""
        if not tab:
            return
            
        # Подсказки для элементов управления на вкладке тестирования  # 常用测试标签页控件的工具提示
        if hasattr(tab, 'test_dataset_path_edit'):
            tab.test_dataset_path_edit.setToolTip("Укажите путь к тестовому набору данных<br><b>Горячая клавиша:</b> нет")  # 指定测试数据集的路径 / 快捷键: 无

        if hasattr(tab, 'model_path_edit'):
            tab.model_path_edit.setToolTip("Укажите путь к файлу весов обученной модели<br><b>Подсказка:</b> выберите подходящие веса модели, соответствующие вашим тестовым данным")  # 指定训练好的模型权重文件路径 / 提示...

        if hasattr(tab, 'conf_threshold_spin'):
            tab.conf_threshold_spin.setToolTip("Порог достоверности детекции<br><b>Подсказка:</b> более высокое значение уменьшает количество ложных срабатываний, но может увеличить число пропусков")  # 检测置信度阈值 / 提示...

        if hasattr(tab, 'start_testing_btn'):
            tab.start_testing_btn.setToolTip("Начать оценку производительности модели<br><b>Убедитесь, что все параметры указаны правильно</b>")  # 开始评估模型性能 / 确保所有参数设置正确

        if hasattr(tab, 'stop_testing_btn'):
            tab.stop_testing_btn.setToolTip("Остановить текущий процесс тестирования<br><b>Предупреждение:</b> после остановки тестирование нельзя будет продолжить")  # 停止当前测试进程 / 警告...

        if hasattr(tab, 'clear_test_log_btn'):
            tab.clear_test_log_btn.setToolTip("Очистить журнал тестирования<br><b>Горячая клавиша:</b> Ctrl+L")  # 清除测试日志 / 快捷键: Ctrl+L
    
    @staticmethod
    def apply_inference_tab_tooltips(tab):
        """为推理标签页添加工具提示"""
        if not tab:
            return
            
        # Подсказки для элементов управления на вкладке инференса  # 常用推理标签页控件的工具提示
        if hasattr(tab, 'inference_source_edit'):
            tab.inference_source_edit.setToolTip("Укажите источник для инференса (изображение, видео или путь к папке)<br><b>Горячая клавиша:</b> нет")  # 指定推理源（图像、视频或文件夹路径） / 快捷键: 无

        if hasattr(tab, 'inference_model_path_edit'):
            tab.inference_model_path_edit.setToolTip("Укажите путь к файлу весов модели для инференса<br><b>Подсказка:</b> выберите подходящую модель для получения наилучших результатов")  # 指定用于推理的模型权重文件路径 / 提示...

        if hasattr(tab, 'start_inference_btn'):
            tab.start_inference_btn.setToolTip("Запустить инференс для выбранного источника<br><b>Убедитесь, что все параметры указаны правильно</b>")  # 开始对选定的源进行推理 / 确保所有参数设置正确

        if hasattr(tab, 'stop_inference_btn'):
            tab.stop_inference_btn.setToolTip("Остановить текущий процесс инференса<br><b>Предупреждение:</b> после остановки инференс нельзя будет продолжить")  # 停止当前推理进程 / 警告...

        if hasattr(tab, 'save_results_btn'):
            tab.save_results_btn.setToolTip("Сохранить результаты инференса<br><b>Подсказка:</b> результаты будут сохранены в указанной выходной папке")  # 保存推理结果 / 提示...
    
    @staticmethod
    def apply_settings_tab_tooltips(tab):
        """为设置标签页添加工具提示"""
        if not tab:
            return
            
        # Подсказки для элементов управления на вкладке настроек  # 常用设置标签页控件的工具提示
        if hasattr(tab, 'device_combo'):
            tab.device_combo.setToolTip("Выберите среду выполнения<br><b>Подсказка:</b> GPU обычно намного быстрее CPU")  # 选择运行环境 / 提示...

        if hasattr(tab, 'output_dir_edit'):
            tab.output_dir_edit.setToolTip("Укажите каталог для сохранения результатов<br><b>Горячая клавиша:</b> нет")  # 指定保存输出结果的目录 / 快捷键: 无

        if hasattr(tab, 'save_settings_btn'):
            tab.save_settings_btn.setToolTip("Сохранить текущие настройки<br><b>Горячая клавиша:</b> Ctrl+S")  # 保存当前设置 / 快捷键: Ctrl+S

        if hasattr(tab, 'reset_settings_btn'):
            tab.reset_settings_btn.setToolTip("Сбросить все настройки до значений по умолчанию<br><b>Предупреждение:</b> это действие нельзя отменить")  # 重置所有设置为默认值 / 警告...

        if hasattr(tab, 'theme_combo') and tab.theme_combo:
            tab.theme_combo.setToolTip("Выберите тему приложения<br><b>Подсказка:</b> тёмная тема помогает снизить нагрузку на глаза")  # 选择应用程序的主题 / 提示...
    
    @staticmethod
    def show_temporary_tooltip(widget, message, duration=3000):
        """显示临时的工具提示"""
        position = widget.mapToGlobal(widget.rect().topRight())
        QToolTip.showText(position, message, widget)
        
        # 创建定时器在指定时间后隐藏工具提示
        QTimer.singleShot(duration, lambda: QToolTip.hideText()) 