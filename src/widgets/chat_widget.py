from PyQt5.QtWidgets import QScrollArea, QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt5.QtGui import QFont

from src.widgets.chat_message import ChatMessage
from styles.colors import Colors
from styles.animations import fade_in, pulse


class ChatWidget(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.messages = []
        self.current_chat_id = None

    def setup_ui(self):
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setContentsMargins(15, 20, 15, 20)
        self.layout.setSpacing(16)
        
        # Empty state message
        self.empty_label = QLabel("No messages yet. Start a conversation!")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setFont(QFont("SF Pro Display, Helvetica Neue, Segoe UI", 14))
        self.layout.addWidget(self.empty_label)
        self.empty_label.hide() # Start hidden and show if needed
        
        self.setWidget(self.container)
        
        # Apply initial style
        self.update_style()

    def update_style(self):
        """Update styling based on current theme colors."""
        # Update scrollbar and background styling
        self.setStyleSheet(f"""
            QScrollArea {{
                background-color: {Colors.BACKGROUND};
                border: none;
            }}
            QScrollBar:vertical {{
                width: 10px;
                background: {Colors.SCROLLBAR_BG};
                border-radius: 4px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {Colors.SCROLLBAR_HANDLE};
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """)
        
        # Update empty label styling
        self.empty_label.setStyleSheet(f"color: {Colors.TEXT}; opacity: 0.7;")
        
        # Update existing chat messages if any
        for i in range(self.layout.count()):
            widget = self.layout.itemAt(i).widget()
            if isinstance(widget, ChatMessage):
                widget.update_style()

    def add_message(self, text, is_user=True):
        """Add a new message to the chat."""
        # Hide empty state if showing
        if self.empty_label.isVisible():
            self.empty_label.hide()
            
        message = ChatMessage(text, is_user)
        self.messages.append(message)
        self.layout.addWidget(message)
        
        # Only use fade_in animation for new messages
        message.animation = fade_in(message, duration=250)
        message.animation.start()
        
        self.scroll_to_bottom()
        return message
        
    def scroll_to_bottom(self):
        """Smooth scroll to the bottom of the chat."""
        scroll_bar = self.verticalScrollBar()
        animation = QPropertyAnimation(scroll_bar, b"value")
        animation.setDuration(300)
        animation.setStartValue(scroll_bar.value())
        animation.setEndValue(scroll_bar.maximum())
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.start()
        
    def clear_messages(self):
        """Clear all messages from the chat."""
        for message in self.messages:
            self.layout.removeWidget(message)
            message.deleteLater()
        
        self.messages.clear()
        self.empty_label.show()
        
    def set_chat(self, chat_id):
        """Set the current chat being displayed."""
        self.current_chat_id = chat_id
        self.clear_messages()
        
    def typing_indicator(self):
        """Show a typing indicator that the AI is generating a response."""
        # Simple typing indicator message
        typing_msg = ChatMessage("...", is_user=False)
        self.layout.addWidget(typing_msg)
        
        # Add a pulse animation
        animation = pulse(typing_msg, scale_factor=1.05, duration=800)
        animation.setLoopCount(-1)  # Loop indefinitely
        animation.start()
        
        self.scroll_to_bottom()
        return typing_msg  # Return so it can be removed when real message arrives