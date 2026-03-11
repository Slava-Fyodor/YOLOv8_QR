import os
from PyQt5.QtWidgets import QSplashScreen, QProgressBar, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import (
    QPixmap, QPainter, QColor, QFont, QLinearGradient,
    QPainterPath, QPen, QBrush
)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QRect, QRectF
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QBrush


class SplashScreen(QSplashScreen):
    def __init__(self):
        # 创建基本pixmap
        pixmap = QPixmap(560, 360)
        pixmap.fill(Qt.transparent)
        super().__init__(pixmap)

        # 设置窗口标志
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 主背景渐变
        self.gradient = QLinearGradient(0, 0, 560, 360)
        self.gradient.setColorAt(0.0, QColor(9, 22, 43))
        self.gradient.setColorAt(0.45, QColor(16, 110, 170))
        self.gradient.setColorAt(1.0, QColor(0, 190, 150))

        # 底部面板渐变
        self.panel_gradient = QLinearGradient(0, 240, 0, 340)
        self.panel_gradient.setColorAt(0.0, QColor(255, 255, 255, 18))
        self.panel_gradient.setColorAt(1.0, QColor(255, 255, 255, 8))

        # 设置进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(42, 294, 476, 12)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 6px;
                background-color: rgba(8, 20, 34, 140);
            }
            QProgressBar::chunk {
                border-radius: 6px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #29D3FF,
                    stop:0.5 #19B7FF,
                    stop:1 #00E39F
                );
            }
        """)

        # 创建状态标签
        self.status_label = QLabel(self)
        self.status_label.setGeometry(42, 266, 476, 18)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setText("Инициализация...")
        self.status_label.setStyleSheet("""
            QLabel {
                color: rgba(235, 243, 250, 220);
                font-size: 12px;
                font-weight: 500;
                background: transparent;
            }
        """)

        # 设置标题字体
        self.setFont(QFont('Segoe UI', 14, QFont.Bold))

        # 创建动画
        self.animation = QPropertyAnimation(self, b"pos")
        self.animation.setDuration(500)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)

    def drawContents(self, painter):
        painter.setRenderHint(QPainter.Antialiasing)

        # 阴影
        shadow_rect = QRectF(12, 14, self.width() - 18, self.height() - 16)
        shadow_path = QPainterPath()
        shadow_path.addRoundedRect(shadow_rect, 24, 24)
        painter.fillPath(shadow_path, QColor(0, 0, 0, 60))

        # 主卡片
        card_rect = QRectF(6, 6, self.width() - 12, self.height() - 12)
        card_path = QPainterPath()
        card_path.addRoundedRect(card_rect, 22, 22)
        painter.fillPath(card_path, QBrush(self.gradient))

        # 高光层
        gloss = QLinearGradient(0, 6, 0, 180)
        gloss.setColorAt(0.0, QColor(255, 255, 255, 40))
        gloss.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.fillPath(card_path, gloss)

        # 边框
        painter.setPen(QPen(QColor(255, 255, 255, 45), 1.2))
        painter.drawPath(card_path)

        # 背景光斑
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 18))
        painter.drawEllipse(-30, -10, 180, 180)

        painter.setBrush(QColor(0, 220, 255, 22))
        painter.drawEllipse(360, 32, 120, 120)

        painter.setBrush(QColor(0, 255, 180, 18))
        painter.drawEllipse(400, 220, 90, 90)

        # 右上角装饰圆环
        cx, cy = 420, 96
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 40), 2))
        painter.drawEllipse(QRect(cx - 38, cy - 38, 76, 76))

        painter.setPen(QPen(QColor(0, 220, 255, 70), 2))
        painter.drawEllipse(QRect(cx - 22, cy - 22, 44, 44))

        painter.setPen(QPen(QColor(255, 255, 255, 60), 1))
        painter.drawLine(cx - 46, cy, cx + 46, cy)
        painter.drawLine(cx, cy - 46, cx, cy + 46)

        # 左侧小标签
        tag_rect = QRectF(36, 34, 74, 26)
        tag_path = QPainterPath()
        tag_path.addRoundedRect(tag_rect, 13, 13)
        painter.fillPath(tag_path, QColor(8, 18, 32, 70))
        painter.setPen(QPen(QColor(255, 255, 255, 35), 1))
        painter.drawPath(tag_path)

        painter.setPen(QColor(230, 240, 248, 220))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(QRect(36, 34, 74, 26), Qt.AlignCenter, "YOLO")

        # 主标题
        painter.setPen(QColor(242, 247, 252))
        painter.setFont(QFont("Segoe UI", 28, QFont.Bold))
        painter.drawText(
            QRect(36, 86, 300, 44),
            Qt.AlignLeft | Qt.AlignVCenter,
            "YOLO Toolkit"
        )

        # 副标题
        painter.setPen(QColor(220, 232, 242, 215))
        painter.setFont(QFont("Segoe UI", 12))
        painter.drawText(
            QRect(38, 132, 320, 22),
            Qt.AlignLeft | Qt.AlignVCenter,
            "Обучение • Валидация • Инференс"
        )

        # 简介文字
        painter.setPen(QColor(225, 236, 245, 150))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(
            QRect(38, 168, 330, 20),
            Qt.AlignLeft | Qt.AlignVCenter,
            "Среда для обучения и инференса YOLO"
        )

        # 分隔线
        painter.setPen(QPen(QColor(255, 255, 255, 26), 1))
        painter.drawLine(36, 212, self.width() - 36, 212)

        # 底部信息面板
        panel_rect = QRectF(24, 238, self.width() - 48, 92)
        panel_path = QPainterPath()
        panel_path.addRoundedRect(panel_rect, 18, 18)
        painter.fillPath(panel_path, QBrush(self.panel_gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
        painter.drawPath(panel_path)

        # 底部辅助提示
        painter.setPen(QColor(225, 235, 244, 135))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(
            QRect(42, 244, 476, 16),
            Qt.AlignCenter,
            "Подготовка модулей и рабочей среды..."
        )

        # 版本徽标
        badge_rect = QRectF(self.width() - 92, 20, 58, 24)
        badge_path = QPainterPath()
        badge_path.addRoundedRect(badge_rect, 12, 12)
        painter.fillPath(badge_path, QColor(9, 20, 35, 80))
        painter.setPen(QPen(QColor(255, 255, 255, 35), 1))
        painter.drawPath(badge_path)

        painter.setPen(QColor(240, 247, 252, 230))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(QRect(self.width() - 92, 20, 58, 24), Qt.AlignCenter, "v3.6.0")

    def updateProgress(self, value, message=""):
        """更新进度条和消息"""
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)

    def showEvent(self, event):
        """显示事件处理"""
        super().showEvent(event)

        # 屏幕中央显示
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        x = screen_geometry.x() + (screen_geometry.width() - self.width()) // 2
        y = screen_geometry.y() + (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        # 轻微上移到落下动画
        self.animation.setStartValue(QPoint(x, y - 20))
        self.animation.setEndValue(QPoint(x, y))
        self.animation.start()


def showSplashScreen(app, main_window):
    """显示启动画面并初始化应用程序"""
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    # 模拟加载过程
    steps = [
        (10, "Инициализация приложения..."),
        (30, "Загрузка компонентов..."),
        (50, "Проверка GPU..."),
        (70, "Загрузка модели..."),
        (90, "Подготовка интерфейса..."),
        (100, "Запуск завершён")
    ]

    for i, (progress, message) in enumerate(steps):
        QTimer.singleShot(i * 300, lambda p=progress, m=message: splash.updateProgress(p, m))

    # 在所有步骤完成后显示主窗口
    QTimer.singleShot(len(steps) * 300 + 200, lambda: finishSplash(splash, main_window))

    return splash


def finishSplash(splash, main_window):
    """完成启动画面并显示主窗口"""
    main_window.show()
    splash.finish(main_window)