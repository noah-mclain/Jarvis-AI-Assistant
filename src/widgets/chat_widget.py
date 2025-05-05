from PyQt5.QtWidgets import QScrollArea, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve

from src.widgets.chat_message import ChatMessage


class ChatWidget(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.messages = []

    def setup_ui(self):
        self.setWidgetResizable(True)
        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setAlignment(Qt.AlignTop)
        self.setWidget(self.container)

    def add_message(self, text, is_user=True):
        message = ChatMessage(text, is_user)
        self.layout.addWidget(message)
        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        scroll_bar = self.verticalScrollBar()
        animation = QPropertyAnimation(scroll_bar, b"value")
        animation.setDuration(400)
        animation.setStartValue(scroll_bar.value())
        animation.setEndValue(scroll_bar.maximum())
        animation.setEasingCurve(QEasingCurve.OutQuad)
        animation.start()