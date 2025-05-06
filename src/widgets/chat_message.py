from PySide6.QtWidgets import (QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, 
                        QFrame, QSizePolicy, QSpacerItem, QMenu)
from PySide6.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QParallelAnimationGroup
from PySide6.QtGui import QPainter, QColor, QLinearGradient, QFont, QIcon, QPixmap, QAction
from styles.colors import Colors
from styles.animations import fade_in, slide_in, scale
import time
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

class ChatMessage(QWidget):
    def __init__(self, text, is_user=True, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.text = text
        self.timestamp = time.strftime("%I:%M %p")
        self.setup_ui(text)
        self.setGraphicsEffect(None)
        
        # Create combined animation for a more fluid appearance
        self.animation_group = QParallelAnimationGroup(self)
        
        # Fade in animation
        fade_animation = fade_in(self, duration=350)
        
        # Subtle slide animation
        slide_animation = slide_in(self, 
                                  direction='up' if not is_user else 'right', 
                                  distance=8, 
                                  duration=350, 
                                  ease=QEasingCurve.OutQuint)
        
        # Combine animations
        self.animation_group.addAnimation(fade_animation)
        self.animation_group.addAnimation(slide_animation)
        self.animation_group.start()

    def setup_ui(self, text):
        self.setMinimumWidth(200)
        self.setMaximumWidth(950)  # Wider to fit more content
        self.setContentsMargins(0, 4, 0, 4)
        
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 4, 20, 4)
        main_layout.setSpacing(14)
        
        # Avatar for user or AI - more compact and rounded
        self.avatar_label = QLabel()
        self.avatar_label.setFixedSize(36, 36)
        self.avatar_label.setScaledContents(True)
        self.avatar_label.setAlignment(Qt.AlignTop)
        self.avatar_label.setObjectName("avatar_label")
        
        # Set avatar images
        if self.is_user:
            self.avatar_pixmap = QPixmap("styles/svg/user_avatar.svg")
        else:
            self.avatar_pixmap = QPixmap("styles/svg/ai_avatar.svg")
            
        self.avatar_label.setPixmap(self.avatar_pixmap)
        
        # Create message container with improved styling
        self.message_container = QFrame()
        self.message_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.message_container.setObjectName("message_container")
        message_layout = QVBoxLayout(self.message_container)
        message_layout.setContentsMargins(0, 0, 0, 0)
        message_layout.setSpacing(4)
        
        # Sender name - more subtle with proper spacing
        name_color = Colors.TEXT if self.is_user else Colors.PRIMARY
        self.sender_label = QLabel("You" if self.is_user else "Jarvis AI")
        self.sender_label.setFont(QFont("SF Pro Display, Helvetica Neue, Segoe UI", 11, QFont.DemiBold))
        self.sender_label.setStyleSheet(f"color: {name_color}; margin-bottom: 1px; opacity: 0.9;")
        
        # Message bubble with improved styling and word wrapping
        self.bubble = QLabel(text)
        self.bubble.setWordWrap(True)
        self.bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.bubble.setCursor(Qt.IBeamCursor)
        self.bubble.setFont(QFont("SF Pro Display, Helvetica Neue, Segoe UI", 14))
        self.bubble.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.bubble.setObjectName("message_bubble")
        self.bubble.setTextFormat(Qt.AutoText)
        self.bubble.setOpenExternalLinks(True)  # Allow links to be opened
        
        # Time label - more subtle and positioned bottom right
        self.time_label = QLabel(self.timestamp)
        self.time_label.setFont(QFont("SF Pro Display, Helvetica Neue, Segoe UI", 9))
        self.time_label.setStyleSheet(f"color: {Colors.TEXT}; opacity: 0.4; margin-top: 2px;")
        
        # Add to container
        message_layout.addWidget(self.sender_label)
        message_layout.addWidget(self.bubble)
        message_layout.addWidget(self.time_label)
        
        # Set correct avatar/message ordering with better spacing
        content_spacing = 80 # Width of the space to create asymmetrical layout
        
        if not self.is_user:
            # AI messages left-aligned
            main_layout.addWidget(self.avatar_label, 0, Qt.AlignTop)
            main_layout.addWidget(self.message_container, 1)
            main_layout.addSpacerItem(QSpacerItem(content_spacing, 0, QSizePolicy.Fixed, QSizePolicy.Minimum))
            self.time_label.setAlignment(Qt.AlignLeft)
        else:
            # User messages right-aligned
            main_layout.addSpacerItem(QSpacerItem(content_spacing, 0, QSizePolicy.Fixed, QSizePolicy.Minimum))
            main_layout.addWidget(self.message_container, 1) 
            main_layout.addWidget(self.avatar_label, 0, Qt.AlignTop)
            self.time_label.setAlignment(Qt.AlignRight)
        
        # Add hover effect
        self.setMouseTracking(True)
        
        # Apply initial style
        self.update_style()

    def update_style(self):
        """Update styling based on current theme colors."""
        # Rounded avatar style
        self.avatar_label.setStyleSheet(f"""
            QLabel#avatar_label {{
                background-color: {Colors.BACKGROUND};
                border-radius: 18px;
                border: none;
                margin-top: 4px;
            }}
        """)
        
        # Message container style
        self.message_container.setStyleSheet(f"""
            QFrame#message_container {{
                background-color: transparent;
                padding: 0px;
            }}
        """)
        
        # Bubble styles with modern appearance
        if self.is_user:
            bubble_bg = Colors.USER_BUBBLE
            text_color = Colors.TEXT
        else:
            bubble_bg = Colors.AI_BUBBLE
            text_color = Colors.TEXT
            
        self.bubble.setStyleSheet(f"""
            QLabel#message_bubble {{
                color: {text_color};
                padding: 16px 18px;
                border-radius: 16px;
                background-color: {bubble_bg};
                margin: 4px 0px;
                line-height: 1.5;
            }}
        """)
        
        # Update container sizing for dynamic content
        self.adjustSize()

    def enterEvent(self, event):
        # Subtle scale animation on hover
        self.scale_animation = scale(self.bubble, end_scale=1.01, duration=100)
        self.scale_animation.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        # Reset scale on mouse leave
        self.scale_animation = scale(self.bubble, end_scale=1.0, duration=100)
        self.scale_animation.start()
        super().leaveEvent(event)
        
    def contextMenuEvent(self, event):
        # Create a custom context menu for interaction
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {Colors.INPUT_BG};
                color: {Colors.TEXT};
                border: 1px solid {Colors.DIVIDER};
                padding: 5px;
                border-radius: 10px;
            }}
            QMenu::item {{
                padding: 8px 24px;
                border-radius: 6px;
                margin: 2px 6px;
            }}
            QMenu::item:selected {{
                background-color: {Colors.SIDEBAR_ITEM_HOVER};
            }}
        """)
        
        # Copy text action
        copy_action = QAction("Copy message", self)
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
            edit_action = QAction("Edit message", self)
            edit_action.triggered.connect(self.edit_message)
            menu.addAction(edit_action)
            
        # Show context menu
        position = event.globalPos()
        menu.exec_(position)
        
    def copy_text(self):
        """Copy message text to clipboard."""
        from PySide6.QtWidgets import QApplication
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
        container_width = self.message_container.width()
        self.bubble.setMaximumWidth(container_width - 24)  # Better padding for text