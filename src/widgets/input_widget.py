from PyQt5.QtWidgets import QWidget, QTextEdit, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont
from styles.colors import Colors

class InputWidget(QWidget):
    message_sent = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Type your message...")
        self.input_field.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.INPUT_BG};
                color: {Colors.TEXT};
                border-radius: 20px;
                padding: 15px;
                font-size: 14px;
            }}
        """)

        self.send_btn = QPushButton()
        self.send_btn.setIcon(QIcon("send_icon.svg"))
        self.send_btn.setIconSize(QSize(24, 24))
        self.send_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.PRIMARY};
                border-radius: 20px;
                padding: 15px;
            }}
            QPushButton:hover {{
                background: {Colors.SECONDARY};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.addWidget(self.input_field, 4)
        layout.addWidget(self.send_btn, 1)
        layout.setContentsMargins(20, 10, 20, 20)

        self.send_btn.clicked.connect(self.send_message)
        self.input_field.textChanged.connect(self.adjust_height)

    def adjust_height(self):
        doc_height = self.input_field.document().size().height()
        self.input_field.setMaximumHeight(min(int(doc_height) + 20, 150))
        
    def send_message(self):
        message = self.input_field.toPlainText().strip()
        if message:
            self.message_sent.emit(message)
            self.input_field.clear()