from PySide6.QtWidgets import QScrollArea, QWidget, QVBoxLayout, QLabel, QSizePolicy, QFrame, QHBoxLayout
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Signal, QParallelAnimationGroup, QByteArray
from PySide6.QtGui import QFont, QColor

from widgets.chat_message import ChatMessage
from styles.colors import Colors
from styles.animations import fade_in, pulse, bounce_in


class ChatWidget(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.messages = []
        self.current_chat_id = None
        self.empty_label = QLabel("No messages yet.")
        
        # Animate entrance
        self.animate_entrance()
        
        self.setup_ui()
        
    def animate_entrance(self):
        """Add entrance animation to the entire chat area."""
        self.setGraphicsEffect(None)  # Clear any existing effects
        
        # Fade in animation
        fade_anim = fade_in(self, duration=400, ease=QEasingCurve.OutCubic)
        fade_anim.start()

    def setup_ui(self):
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFrameShape(QFrame.NoFrame)  # Remove border
        
        self.container = QWidget()
        self.container.setObjectName("chat_container")
        
        self.layout = QVBoxLayout(self.container)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setContentsMargins(15, 20, 15, 20)
        self.layout.setSpacing(18)  # Increased spacing between messages
        
        # Empty state message with nicer styling
        self.empty_label = QLabel("No messages yet. Start a conversation!")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setFont(QFont("SF Pro Display, Helvetica Neue, Segoe UI", 15))
        self.empty_label.setObjectName("empty_label")
        self.layout.addWidget(self.empty_label)
        self.empty_label.hide() # Start hidden and show if needed
        
        self.setWidget(self.container)
        
        # Apply initial style
        self.update_style()
        
    def update_style(self):
        """Update styling based on current theme colors."""
        # Set chat area style
        self.setStyleSheet(f"""
            ChatWidget {{
                background-color: {Colors.BACKGROUND};
                border: none;
            }}
            
            QWidget#chat_container {{
                background-color: {Colors.BACKGROUND};
            }}
            
            QLabel#empty_label {{
                color: rgba({self._hexToRgb(Colors.TEXT)}, 0.5);
                font-weight: 300;
                padding: 40px;
                margin: 60px 0px;
            }}
            
            QScrollBar:vertical {{
                width: 5px;
                background: rgba(0, 0, 0, 0);
                border-radius: 2px;
                margin: 0px;
            }}
            
            QScrollBar::handle:vertical {{
                background: rgba(255, 255, 255, 0.12);
                border-radius: 2px;
                min-height: 40px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background: rgba(255, 255, 255, 0.20);
            }}
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """)
        
        # Update all message widgets
        for message in self.messages:
            message.update_style()
            
    def _hexToRgb(self, hex_color):
        """Convert hex color to RGB for opacity support."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"{r}, {g}, {b}"
        
    def add_message(self, text, is_user=True):
        """Add a message to the chat with smooth animation."""
        # Hide empty label if it's visible
        if self.empty_label.isVisible():
            # Fade out the empty label
            fade_out = QPropertyAnimation(self.empty_label, "windowOpacity")
            fade_out.setDuration(200)
            fade_out.setStartValue(1.0)
            fade_out.setEndValue(0.0)
            fade_out.setEasingCurve(QEasingCurve.InCubic)
            
            def hide_label():
                self.empty_label.hide()
            
            fade_out.finished.connect(hide_label)
            fade_out.start()
        
        # Create message
        message = ChatMessage(text, is_user)
        self.layout.addWidget(message)
        self.messages.append(message)
        
        # Scroll to the new message with animation
        QTimer.singleShot(50, self.smooth_scroll_to_bottom)
        
        return message
        
    def scroll_to_bottom(self):
        """Smooth scroll to the bottom of the chat."""
        self.smooth_scroll_to_bottom()
        
    def smooth_scroll_to_bottom(self, duration=300):
        """Enhanced smooth scrolling with easing."""
        scroll_bar = self.verticalScrollBar()
        current = scroll_bar.value()
        maximum = scroll_bar.maximum()
        
        # Only animate if there's a significant difference
        if maximum - current > 10:
            animation = QPropertyAnimation(scroll_bar, QByteArray(b"value"))
            animation.setDuration(duration)
            animation.setStartValue(current)
            animation.setEndValue(maximum)
            animation.setEasingCurve(QEasingCurve.OutCubic)
            animation.start()
        else:
            # Small difference, just jump
            scroll_bar.setValue(maximum)
        
    def clear_messages(self):
        """Clear all messages from the chat with animation."""
        if not self.messages:
            return
            
        # Create fade out animations for all messages
        for message in self.messages:
            fade_out = QPropertyAnimation(message, "windowOpacity")
            fade_out.setDuration(200)
            fade_out.setStartValue(1.0)
            fade_out.setEndValue(0.0)
            fade_out.setEasingCurve(QEasingCurve.InCubic)
            fade_out.start()
        
        # After animation, remove all messages
        def remove_all():
            for message in self.messages:
                self.layout.removeWidget(message)
                message.deleteLater()
            
            self.messages.clear()
            self.empty_label.show()
            
            # Animate empty label appearing
            fade_in_anim = fade_in(self.empty_label, duration=300)
            bounce_anim = bounce_in(self.empty_label, direction='up', distance=20, duration=500)
            
            anim_group = QParallelAnimationGroup(self.empty_label)
            anim_group.addAnimation(fade_in_anim)
            anim_group.addAnimation(bounce_anim)
            anim_group.start()
            
        # Wait for fade out to complete
        QTimer.singleShot(250, remove_all)
        
    def set_chat(self, chat_id):
        """Set the current chat being displayed."""
        self.current_chat_id = chat_id
        self.clear_messages()
        
    def typing_indicator(self):
        """Show a typing indicator that the AI is generating a response."""
        # Create a nicer typing indicator
        typing_container = QWidget()
        typing_container.setObjectName("typing_indicator")
        typing_layout = QHBoxLayout(typing_container)
        typing_layout.setContentsMargins(20, 10, 20, 10)
        
        # Create typing dots
        dots = QLabel("...")
        dots.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        dots.setFont(QFont("SF Pro Display, Helvetica Neue, Segoe UI", 16, QFont.Bold))
        dots.setStyleSheet(f"color: {Colors.PRIMARY}; margin-left: 10px;")
        
        typing_layout.addWidget(dots)
        
        # Add to layout
        self.layout.addWidget(typing_container)
        
        # Style the typing indicator
        typing_container.setStyleSheet(f"""
            QWidget#typing_indicator {{
                background-color: {Colors.AI_BUBBLE};
                border-radius: 16px;
                max-width: 100px;
                margin-left: 60px;
            }}
        """)
        
        # Add a pulse animation
        animation = pulse(dots, scale_factor=1.1, duration=600)
        animation.setLoopCount(-1)  # Loop indefinitely
        animation.start()
        
        self.smooth_scroll_to_bottom(200)  # Faster scroll for typing indicator
        return typing_container