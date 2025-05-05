import sys
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QPushButton, QLabel,
                             QScrollArea, QFrame)
from PyQt5.QtGui import QFont, QTextCursor, QIcon, QPalette, QColor

class ChatMessage(QWidget):
    def __init__(self, text, is_user=True, parent=None):
        super().__init__(parent)
        self.setup_ui(text, is_user)
        
    def setup_ui(self, text, is_user):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: %s;
                border-radius: 15px;
                padding: 10px;
                margin: 5px;
            }
        """ % ("#2b5278" if is_user else "#1f4b34"))
        
        frame_layout = QVBoxLayout(frame)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("color: white; font-size: 12pt;")
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        frame_layout.addWidget(self.label)
        self.layout.addWidget(frame)
        self.setMaximumWidth(600)
        self.setMinimumWidth(200)
        
        if is_user:
            frame_layout.setAlignment(Qt.AlignRight)
        else:
            frame_layout.setAlignment(Qt.AlignLeft)

class ChatWidget(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        self.container = QWidget()
        self.container.setStyleSheet("background-color: #1a1a1a;")
        self.layout = QVBoxLayout(self.container)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        self.setWidget(self.container)
        
    def add_message(self, text, is_user=True):
        message = ChatMessage(text, is_user)
        self.layout.addWidget(message)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class InputWidget(QWidget):
    message_sent = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Type your message...")
        self.input_field.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: white;
                border: 2px solid #3d3d3d;
                border-radius: 15px;
                padding: 10px;
                font-size: 12pt;
                min-height: 50px;
                max-height: 150px;
            }
        """)
        self.input_field.setFocus()
        
        self.send_btn = QPushButton()
        self.send_btn.setIcon(QIcon.fromTheme("mail-send"))
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #2b5278;
                border: none;
                border-radius: 15px;
                min-width: 60px;
                max-width: 60px;
                min-height: 50px;
            }
            QPushButton:hover {
                background-color: #36638d;
            }
        """)
        self.send_btn.clicked.connect(self.send_message)
        
        self.layout.addWidget(self.input_field, 4)
        self.layout.addWidget(self.send_btn, 1)
        
    def send_message(self):
        if text := self.input_field.toPlainText().strip():
            self.message_sent.emit(text)
            self.input_field.clear()
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return and not event.modifiers() & Qt.ShiftModifier:
            self.send_message()
            event.accept()
        else:
            super().keyPressEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        self.setWindowTitle("JARVIS AI Assistant")
        self.setMinimumSize(800, 600)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        self.layout = QVBoxLayout(main_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        self.chat_widget = ChatWidget()
        self.input_widget = InputWidget()
        
        self.layout.addWidget(self.chat_widget, 4)
        self.layout.addWidget(self.input_widget, 1)
        
        self.setStyleSheet("background-color: #1a1a1a;")
        
    def setup_connections(self):
        self.input_widget.message_sent.connect(self.add_user_message)
        
    def add_user_message(self, text):
        self.chat_widget.add_message(text, is_user=True)
        # Here you would call your AI processing code
        # For example: response = ai.process(text)
        # Then call self.add_ai_message(response)
        
    def add_ai_message(self, text):
        self.chat_widget.add_message(text, is_user=False)
        
    def add_error_message(self, text):
        error_message = ChatMessage(f"Error: {text}", is_user=False)
        error_message.label.setStyleSheet("color: #ff4444; font-size: 12pt;")
        self.chat_widget.layout.addWidget(error_message)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(26, 26, 26))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())