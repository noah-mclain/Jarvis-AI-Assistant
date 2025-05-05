from PyQt5.QtWidgets import QWidget, QTextEdit, QPushButton, QHBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QEvent
from PyQt5.QtGui import QIcon, QFont, QKeyEvent
from styles.colors import Colors
from styles.animations import scale

class InputWidget(QWidget):
    message_sent = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.installEventFilter(self)

    def setup_ui(self):
        self.setFixedHeight(100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Input field
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Type your message...")
        self.input_field.setFont(QFont("SF Pro Display, Helvetica Neue, Segoe UI", 12))
        self.input_field.setAcceptRichText(False)
        self.input_field.setTabChangesFocus(True)
        self.input_field.setMinimumHeight(60)
        self.input_field.setMaximumHeight(100)

        # Send button
        self.send_btn = QPushButton()
        self.send_btn.setIcon(QIcon("styles/svg/send_icon.svg"))
        self.send_btn.setIconSize(QSize(24, 24))
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.setFixedSize(60, 60)

        # Layout
        layout = QHBoxLayout(self)
        layout.addWidget(self.input_field, 1)
        layout.addWidget(self.send_btn)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(15)

        # Connect signals
        self.send_btn.clicked.connect(self.send_message)
        self.input_field.textChanged.connect(self.adjust_height)
        
        # Add hover animation for the send button
        self.send_btn.enterEvent = self.button_enter_event
        self.send_btn.leaveEvent = self.button_leave_event
        
        # Apply initial style
        self.update_style()

    def update_style(self):
        """Update styling based on current theme colors."""
        # Update widget style
        self.setStyleSheet(f"""
            InputWidget {{
                background-color: {Colors.BACKGROUND};
                border-top: 1px solid {Colors.DIVIDER};
                padding: 10px 0px;
            }}
        """)
        
        # Update input field style
        self.input_field.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.INPUT_BG};
                color: {Colors.TEXT};
                border-radius: 20px;
                padding: 12px 15px;
                border: 1px solid transparent;
            }}
            QTextEdit:focus {{
                border: 1px solid {Colors.SECONDARY};
            }}
        """)
        
        # Update send button style
        self.send_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.PRIMARY};
                border-radius: 30px;
                padding: 12px;
            }}
            QPushButton:hover {{
                background: {Colors.SECONDARY};
            }}
            QPushButton:pressed {{
                background: {Colors.PRIMARY};
            }}
        """)

    def adjust_height(self):
        """Adjust input field height based on content."""
        doc_height = self.input_field.document().size().height()
        new_height = min(max(60, int(doc_height) + 24), 100)
        self.input_field.setMinimumHeight(new_height)
        self.input_field.setMaximumHeight(new_height)
        
    def button_enter_event(self, event):
        """Handle send button hover animation."""
        animation = scale(self.send_btn, end_scale=1.05, duration=150)
        animation.start()
        
    def button_leave_event(self, event):
        """Handle send button hover animation exit."""
        animation = scale(self.send_btn, end_scale=1.0, duration=150)
        animation.start()
        
    def send_message(self):
        """Send the message in the input field."""
        if message := self.input_field.toPlainText().strip():
            self.message_sent.emit(message)
            self.input_field.clear()
            self.input_field.setFocus()
    
    def eventFilter(self, obj, event):
        """Handle Shift+Enter for new line and Enter for sending message."""
        if obj is self and event.type() == QEvent.KeyPress:
            key_event = QKeyEvent(event)
            if key_event.key() in [Qt.Key_Return, Qt.Key_Enter]:
                if key_event.modifiers() & Qt.ShiftModifier:
                    return False
                self.send_message()
                return True
        return super().eventFilter(obj, event)
        
    def resizeEvent(self, event):
        """Handle widget resize event."""
        super().resizeEvent(event)
        # Ensure input field width adjusts properly
        self.adjustSize()