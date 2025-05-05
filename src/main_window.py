from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from src.widgets.chat_widget import ChatWidget
from src.widgets.input_widget import InputWidget
from styles.colors import Colors

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setWindowTitle("JARVIS AI")
        self.setMinimumSize(800, 600)
        self.setStyleSheet(f"background-color: {Colors.BACKGROUND};")

    def setup_ui(self):
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.chat_widget = ChatWidget()
        self.input_widget = InputWidget()

        layout.addWidget(self.chat_widget)
        layout.addWidget(self.input_widget)

        self.input_widget.message_sent.connect(self.add_user_message)
        self.setCentralWidget(central_widget)

    def add_user_message(self, text):
        self.chat_widget.add_message(text, is_user=True)
        # Add AI response logic here
        self.add_ai_message("This is a sample response")

    def add_ai_message(self, text):
        self.chat_widget.add_message(text, is_user=False)