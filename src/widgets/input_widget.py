from PySide6.QtWidgets import (QWidget, QTextEdit, QPushButton, QHBoxLayout, 
                         QVBoxLayout, QSizePolicy, QFrame, QLabel)
from PySide6.QtCore import Qt, Signal, QSize, QEvent, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, QTimer, QByteArray
from PySide6.QtGui import QIcon, QFont, QKeyEvent, QColor, QTextCursor
from styles.colors import Colors
from styles.animations import scale, fade_in, slide_in, bounce_in, ripple_effect

class FloatingButton(QPushButton):
    def __init__(self, icon_path, tooltip, parent=None):
        super().__init__(parent)
        self.setIcon(QIcon(icon_path))
        self.setIconSize(QSize(20, 20))
        self.setToolTip(tooltip)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(40, 40)
        self.setObjectName("floating_button")
        
        # Set initial opacity to 0 for entrance animation
        self.setGraphicsEffect(None)
        
    def enterEvent(self, event):
        """Handle button hover animation with improved visuals."""
        anim_group = QParallelAnimationGroup(self)
        
        # Scale up slightly
        scale_anim = scale(self, end_scale=1.08, duration=150, ease=QEasingCurve.OutQuad)
        anim_group.addAnimation(scale_anim)
        anim_group.start()
        
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """Handle button hover animation exit."""
        anim_group = QParallelAnimationGroup(self)
        
        # Scale back to normal
        scale_anim = scale(self, end_scale=1.0, duration=150, ease=QEasingCurve.OutQuad)
        anim_group.addAnimation(scale_anim)
        anim_group.start()
        
        super().leaveEvent(event)
        
    def mousePressEvent(self, event):
        """Handle press with visual feedback."""
        # Scale down on press
        scale_anim = scale(self, end_scale=0.95, duration=100, ease=QEasingCurve.OutQuad)
        scale_anim.start()
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle release with visual feedback."""
        # Scale back to normal
        scale_anim = scale(self, end_scale=1.0, duration=150, ease=QEasingCurve.OutQuad)
        scale_anim.start()
        super().mouseReleaseEvent(event)

class InputWidget(QWidget):
    message_sent = Signal(str)
    
    # New signals for specialized generation types
    story_generation = Signal()
    image_generation = Signal()
    code_generation = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.installEventFilter(self)
        self.generation_mode = None
        
        # Entrance animation
        self.animate_entrance()

    def animate_entrance(self):
        """Add entrance animation to the input widget."""
        anim_group = QParallelAnimationGroup(self)
        
        # Fade in
        fade = fade_in(self, duration=500, ease=QEasingCurve.OutCubic)
        
        # Slide up slightly
        slide = slide_in(self, direction='up', distance=15, duration=500, ease=QEasingCurve.OutQuint)
        
        anim_group.addAnimation(fade)
        anim_group.addAnimation(slide)
        anim_group.start()

    def setup_ui(self):
        self.setFixedHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 10, 20, 20)
        main_layout.setSpacing(8)
        
        # Generation button container
        self.button_container = QFrame()
        self.button_container.setObjectName("button_container")
        button_layout = QHBoxLayout(self.button_container)
        button_layout.setContentsMargins(0, 0, 0, 8)
        button_layout.setSpacing(14)
        button_layout.setAlignment(Qt.AlignCenter)
        
        # Create floating buttons for different generation types
        self.code_btn = FloatingButton("src/styles/svg/code_icon.svg", "Generate Code")
        self.story_btn = FloatingButton("src/styles/svg/story_icon.svg", "Generate Story")
        self.image_btn = FloatingButton("src/styles/svg/image_icon.svg", "Generate Image")
        
        # Add the buttons to the layout
        button_layout.addWidget(self.code_btn)
        button_layout.addWidget(self.story_btn)
        button_layout.addWidget(self.image_btn)
        
        # Input and send container
        input_container = QFrame()
        input_container.setObjectName("input_container")
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(15)
        
        # Input field with modern styling
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Ask me anything...")
        self.input_field.setFont(QFont("SF Pro Display, Helvetica Neue, Segoe UI", 14))
        self.input_field.setAcceptRichText(False)
        self.input_field.setTabChangesFocus(True)
        self.input_field.setMinimumHeight(60)
        self.input_field.setMaximumHeight(100)
        self.input_field.setObjectName("input_field")
        
        # Generation mode indicator
        self.mode_indicator = QLabel("")
        self.mode_indicator.hide()
        self.mode_indicator.setObjectName("mode_indicator")
        input_layout.addWidget(self.mode_indicator)
        
        # Send button with modern styling
        self.send_btn = QPushButton()
        self.send_btn.setIcon(QIcon("src/styles/svg/send_icon.svg"))
        self.send_btn.setIconSize(QSize(24, 24))
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.setFixedSize(50, 50)
        self.send_btn.setObjectName("send_btn")
        
        input_layout.addWidget(self.input_field, 1)
        input_layout.addWidget(self.send_btn)
        
        # Add components to main layout
        main_layout.addWidget(self.button_container)
        main_layout.addWidget(input_container)
        
        # Connect signals
        self.send_btn.clicked.connect(self.send_message)
        self.input_field.textChanged.connect(self.adjust_height)
        
        # Connect generation buttons
        self.code_btn.clicked.connect(self.set_code_generation_mode)
        self.story_btn.clicked.connect(self.set_story_generation_mode)
        self.image_btn.clicked.connect(self.set_image_generation_mode)
        
        # Apply initial style
        self.update_style()
        
        # Delayed button animation (staggered)
        self.animate_buttons()
        
    def animate_buttons(self):
        """Add staggered animation to the feature buttons."""
        # Set initial state for animations
        delay = 100
        duration = 400
        
        # Animate each button with a delay
        for i, btn in enumerate([self.code_btn, self.story_btn, self.image_btn]):
            # Combine animations
            btn_anim_group = QParallelAnimationGroup(btn)
            
            # Fade in
            fade = fade_in(btn, duration=duration)
            
            # Slide in from bottom
            slide = slide_in(btn, direction='up', distance=20, duration=duration, ease=QEasingCurve.OutQuint)
            
            # Add to group
            btn_anim_group.addAnimation(fade)
            btn_anim_group.addAnimation(slide)
            
            # Start with delay
            QTimer.singleShot(delay * i, btn_anim_group.start)

    def update_style(self):
        """Update styling based on current theme colors."""
        # Update widget style
        self.setStyleSheet(f"""
            InputWidget {{
                background-color: {Colors.BACKGROUND};
                border-top: 1px solid {Colors.DIVIDER};
                padding: 10px 0px;
            }}
            
            #button_container {{
                background-color: transparent;
            }}
            
            #floating_button {{
                background: {Colors.BUTTON_BG};
                border-radius: 20px;
                padding: 8px;
                margin: 2px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
            }}
            
            #floating_button:hover {{
                background: {Colors.BUTTON_HOVER};
                box-shadow: 0 3px 8px rgba(0, 0, 0, 0.25);
                transition: all 0.2s ease;
            }}
            
            #mode_indicator {{
                color: {Colors.PRIMARY};
                background-color: {Colors.INPUT_BG};
                padding: 4px 10px;
                border-radius: 12px;
                min-width: 120px;
                max-width: 150px;
                font-size: 12px;
                font-weight: bold;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }}
            
            #input_container {{
                background: transparent;
            }}
        """)
        
        # Update input field style
        self.input_field.setStyleSheet(f"""
            QTextEdit#input_field {{
                background: {Colors.INPUT_BG};
                color: {Colors.TEXT};
                border-radius: 24px;
                padding: 15px 20px;
                border: 1px solid {Colors.DIVIDER};
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            }}
            QTextEdit#input_field:focus {{
                border: 1px solid {Colors.SECONDARY};
                background: {Colors.INPUT_BG};
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }}
        """)
        
        # Update send button style
        self.send_btn.setStyleSheet(f"""
            QPushButton#send_btn {{
                background: {Colors.PRIMARY};
                border-radius: 25px;
                padding: 12px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
            }}
            QPushButton#send_btn:hover {{
                background: {Colors.SECONDARY};
                box-shadow: 0 3px 8px rgba(0, 0, 0, 0.25);
                transform: translateY(-1px);
                transition: all 0.2s ease;
            }}
            QPushButton#send_btn:pressed {{
                background: {Colors.PRIMARY};
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
                transform: translateY(1px);
            }}
        """)

    def adjust_height(self):
        """Adjust input field height based on content with smoother animation."""
        doc_height = self.input_field.document().size().height()
        new_height = min(max(60, int(doc_height) + 24), 100)
        
        # Use animation to smoothly adjust height
        current_height = self.input_field.height()
        if abs(current_height - new_height) > 5:  # Only animate significant changes
            height_anim = QPropertyAnimation(self.input_field, "minimumHeight")
            height_anim.setDuration(100)
            height_anim.setStartValue(current_height)
            height_anim.setEndValue(new_height)
            height_anim.setEasingCurve(QEasingCurve.OutQuad)
            height_anim.start()
            
            # Match maximum height to minimum
            max_anim = QPropertyAnimation(self.input_field, "maximumHeight")
            max_anim.setDuration(100)
            max_anim.setStartValue(current_height)
            max_anim.setEndValue(new_height)
            max_anim.setEasingCurve(QEasingCurve.OutQuad)
            max_anim.start()
        else:
            # Small changes can be instant
            self.input_field.setMinimumHeight(new_height)
            self.input_field.setMaximumHeight(new_height)
    
    def set_code_generation_mode(self):
        """Set the input field to code generation mode with improved animation."""
        self.generation_mode = "code"
        
        # Clear field first
        self.input_field.clear()
        
        # Prepare placeholder and text
        self.input_field.setPlaceholderText("What code would you like me to write?")
        self.input_field.setText("write the code/algorithm for ")
        
        # Focus and move cursor to end of text
        self.input_field.setFocus()
        cursor = self.input_field.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.input_field.setTextCursor(cursor)
        
        # Update and animate the mode indicator
        self.mode_indicator.setText("Code Generation")
        self._animate_mode_indicator()
        
        # Emit signal
        self.code_generation.emit()
    
    def set_story_generation_mode(self):
        """Set the input field to story generation mode with improved animation."""
        self.generation_mode = "story"
        
        # Clear field first
        self.input_field.clear()
        
        # Prepare placeholder and text
        self.input_field.setPlaceholderText("What story would you like me to write?")
        self.input_field.setText("write a story about ")
        
        # Focus and move cursor to end of text
        self.input_field.setFocus()
        cursor = self.input_field.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.input_field.setTextCursor(cursor)
        
        # Update and animate the mode indicator
        self.mode_indicator.setText("Story Generation")
        self._animate_mode_indicator()
        
        # Emit signal
        self.story_generation.emit()
    
    def set_image_generation_mode(self):
        """Set the input field to image generation mode with improved animation."""
        self.generation_mode = "image"
        
        # Clear field first
        self.input_field.clear()
        
        # Prepare placeholder and text
        self.input_field.setPlaceholderText("What image would you like me to generate?")
        self.input_field.setText("generate an image about ")
        
        # Focus and move cursor to end of text
        self.input_field.setFocus()
        cursor = self.input_field.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.input_field.setTextCursor(cursor)
        
        # Update and animate the mode indicator
        self.mode_indicator.setText("Image Generation")
        self._animate_mode_indicator()
        
        # Emit signal
        self.image_generation.emit()
    
    def _animate_mode_indicator(self):
        """Animate the mode indicator appearance with improved effects."""
        # Clear any previous animation effects
        self.mode_indicator.setGraphicsEffect(None)
        
        # First show the indicator
        self.mode_indicator.show()
        
        # Create animation group for combined effects
        anim_group = QParallelAnimationGroup(self.mode_indicator)
        
        # Add fade-in animation
        fade_anim = fade_in(self.mode_indicator, duration=250)
        
        # Add bounce animation
        bounce_anim = bounce_in(self.mode_indicator, direction='left', distance=15, duration=400)
        
        # Add animations to group
        anim_group.addAnimation(fade_anim)
        anim_group.addAnimation(bounce_anim)
        
        # Start combined animation
        anim_group.start()
        
    def send_message(self):
        """Send the message in the input field with visual feedback."""
        if message := self.input_field.toPlainText().strip():
            # Add a small ripple effect on the send button
            ripple = scale(self.send_btn, end_scale=1.2, duration=150, ease=QEasingCurve.OutQuad)
            ripple.start()
            
            # Emit the message
            self.message_sent.emit(message)
            
            # Clear the input field
            self.input_field.clear()
            self.input_field.setFocus()
            
            # If we were in a special mode, animate the mode indicator disappearing
            if self.generation_mode:
                # Fade out the mode indicator
                fade_out = QPropertyAnimation(self.mode_indicator, QByteArray(b"windowOpacity"))
                fade_out.setDuration(200)
                fade_out.setStartValue(1.0)
                fade_out.setEndValue(0.0)
                fade_out.setEasingCurve(QEasingCurve.InCubic)
                
                # When animation finishes, hide the indicator and reset mode
                def finish_reset():
                    self.mode_indicator.hide()
                    self.generation_mode = None
                    self.input_field.setPlaceholderText("Ask me anything...")
                
                fade_out.finished.connect(finish_reset)
                fade_out.start()
    
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
        """Handle widget resize event for responsive layout."""
        super().resizeEvent(event)
        # Ensure input container scales properly
        self.adjustSize()
        
        # Ensure buttons stay centered
        self.button_container.adjustSize()