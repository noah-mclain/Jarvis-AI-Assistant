from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPainter, QColor, QLinearGradient
from styles.colors import Colors
from styles.animations import fade_in, slide_in

class ChatMessage(QWidget):
    def __init__(self, text, is_user=True, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.setup_ui(text)
        self.setGraphicsEffect(None)
        self.animation = fade_in(self)
        self.animation.start()

    def setup_ui(self, text):
        self.setMinimumWidth(200)
        self.setMaximumWidth(600)
        self.setContentsMargins(20, 10, 20, 10)

        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT};
                font-size: 14px;
                padding: 12px;
                border-radius: 15px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {Colors.USER_BUBBLE if self.is_user else Colors.AI_BUBBLE},
                    stop:1 {Colors.SECONDARY if self.is_user else Colors.PRIMARY}
                );
            }}
        """)

        layout = QHBoxLayout(self)
        layout.addWidget(self.label)
        layout.setAlignment(Qt.AlignRight if self.is_user else Qt.AlignLeft)

    def enterEvent(self, event):
        self.scale_animation = QPropertyAnimation(self, b"geometry")
        self.scale_animation.setDuration(200)
        geo = self.geometry()
        center = geo.center()
        new_width = int(geo.width() * 1.02)
        new_height = int(geo.height() * 1.02)
        new_geo = geo
        new_geo.setWidth(new_width)
        new_geo.setHeight(new_height)
        new_geo.moveCenter(center)
        self.scale_animation.setEndValue(new_geo)
        self.scale_animation.start()

    def leaveEvent(self, event):
        self.scale_animation = QPropertyAnimation(self, b"geometry")
        self.scale_animation.setDuration(200)
        self.scale_animation.setEndValue(self.geometry())
        self.scale_animation.start()