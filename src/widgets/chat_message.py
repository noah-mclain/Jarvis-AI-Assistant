from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PyQt5.QtGui import QPainter, QColor, QLinearGradient, QFont, QIcon
from styles.colors import Colors
from styles.animations import fade_in, slide_in, scale

class ChatMessage(QWidget):
    def __init__(self, text, is_user=True, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.text = text
        self.setup_ui(text)
        self.setGraphicsEffect(None)
        self.animation = fade_in(self)
        self.animation.start()

    def setup_ui(self, text):
        self.setMinimumWidth(200)
        self.setMaximumWidth(800)  # Wider to fit more content
        self.setContentsMargins(0, 5, 0, 5)
        
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create container for message with correct alignment
        self.container = QWidget()
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(5)
        
        # Message bubble
        self.bubble = QLabel(text)
        self.bubble.setWordWrap(True)
        self.bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.bubble.setCursor(Qt.IBeamCursor)
        self.bubble.setFont(QFont("SF Pro Display, Helvetica Neue, Segoe UI", 12))
        
        # Add to container
        container_layout.addWidget(self.bubble)
        
        # Add timestamp or other features below the bubble
        self.time_label = QLabel("Just now")
        self.time_label.setFont(QFont("SF Pro Display, Helvetica Neue, Segoe UI", 9))
        self.time_label.setAlignment(Qt.AlignRight if self.is_user else Qt.AlignLeft)
        container_layout.addWidget(self.time_label)
        
        # Set alignment based on user vs AI
        main_layout.addStretch(1 if self.is_user else 0)
        main_layout.addWidget(self.container)
        main_layout.addStretch(0 if self.is_user else 1)
        
        # Add hover effect
        self.setMouseTracking(True)
        
        # Apply initial style
        self.update_style()

    def update_style(self):
        """Update styling based on current theme colors."""
        # Determine bubble colors based on user/AI and current theme
        bubble_color = Colors.USER_BUBBLE if self.is_user else Colors.AI_BUBBLE
        gradient_color = Colors.SECONDARY if self.is_user else Colors.PRIMARY
        text_padding = 15
        
        # Update bubble style
        self.bubble.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT};
                padding: {text_padding}px;
                border-radius: 18px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {bubble_color},
                    stop:1 {gradient_color}
                );
            }}
        """)
        
        # Update timestamp label
        self.time_label.setStyleSheet(f"color: {Colors.TEXT}; opacity: 0.6;")
        
        # Update container sizing for dynamic content
        self.adjustSize()

    def enterEvent(self, event):
        # Subtle scale animation on hover
        self.scale_animation = scale(self.bubble, end_scale=1.02, duration=150)
        self.scale_animation.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        # Reset scale on mouse leave
        self.scale_animation = scale(self.bubble, end_scale=1.0, duration=150)
        self.scale_animation.start()
        super().leaveEvent(event)
        
    def contextMenuEvent(self, event):
        # Create a custom context menu for interaction
        from PyQt5.QtWidgets import QMenu, QAction
        
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {Colors.INPUT_BG};
                color: {Colors.TEXT};
                border: 1px solid {Colors.DIVIDER};
                padding: 5px;
            }}
            QMenu::item {{
                padding: 5px 20px;
            }}
            QMenu::item:selected {{
                background-color: {Colors.SIDEBAR_ITEM_HOVER};
            }}
        """)
        
        # Copy text action
        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self.copy_text)
        menu.addAction(copy_action)
        
        # Additional actions depending on user or AI
        if not self.is_user:
            # For AI messages
            regenerate_action = QAction("Regenerate response", self)
            regenerate_action.triggered.connect(self.regenerate_response)
            menu.addAction(regenerate_action)
        else:
            # For user messages
            edit_action = QAction("Edit", self)
            edit_action.triggered.connect(self.edit_message)
            menu.addAction(edit_action)
            
        # Show context menu
        menu.exec_(event.globalPos())
        
    def copy_text(self):
        """Copy message text to clipboard."""
        from PyQt5.QtWidgets import QApplication
        QApplication.clipboard().setText(self.text)
        
    def regenerate_response(self):
        """Signal to regenerate AI response."""
        # This would connect to the parent widget's regenerate function
        parent = self.parent()
        while parent and not hasattr(parent, 'regenerate_response'):
            parent = parent.parent()
            
        if parent and hasattr(parent, 'regenerate_response'):
            parent.regenerate_response()
            
    def edit_message(self):
        """Allow editing of user message."""
        # This would connect to the parent widget's edit function
        parent = self.parent()
        while parent and not hasattr(parent, 'edit_user_message'):
            parent = parent.parent()
            
        if parent and hasattr(parent, 'edit_user_message'):
            parent.edit_user_message(self, self.text)
            
    def resizeEvent(self, event):
        """Handle widget resize event."""
        super().resizeEvent(event)
        # Adjust bubble width to fit container
        self.bubble.setMaximumWidth(self.width() - 40)  # Add some padding