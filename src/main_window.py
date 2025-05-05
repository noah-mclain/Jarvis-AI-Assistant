from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                          QSplitter, QSizePolicy, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSlot, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QIcon, QFont, QTransform, QPainter, QPen
from src.widgets.chat_widget import ChatWidget
from src.widgets.input_widget import InputWidget
from src.widgets.sidebar_widget import SidebarWidget
from src.widgets.settings_dialog import SettingsDialog
from styles.colors import Colors
from styles.animations import slide_in, slide_out, fade_in, combo_animation
import uuid
import time

class AnimatedToggleButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("toggle_sidebar_btn")
        self.setFixedSize(40, 40)
        self.setCursor(Qt.PointingHandCursor)
        self.is_active = False
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                padding: 8px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {Colors.SIDEBAR_ITEM_HOVER};
            }}
        """)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set pen properties
        pen = QPen(Qt.white)
        pen.setWidth(2)
        painter.setPen(pen)
        
        width, height = self.width(), self.height()
        center_x, center_y = width // 2, height // 2
        
        if not self.is_active:
            # Draw three bars
            bar_length = int(width * 0.5)
            bar_spacing = int(height * 0.2)
            
            # Top bar
            painter.drawLine(
                int(center_x - bar_length / 2), int(center_y - bar_spacing),
                int(center_x + bar_length / 2), int(center_y - bar_spacing)
            )
            
            # Middle bar
            painter.drawLine(
                int(center_x - bar_length / 2), center_y,
                int(center_x + bar_length / 2), center_y
            )
            
            # Bottom bar
            painter.drawLine(
                int(center_x - bar_length / 2), int(center_y + bar_spacing),
                int(center_x + bar_length / 2), int(center_y + bar_spacing)
            )
        else:
            # Draw X
            x_size = int(width * 0.4)
            painter.drawLine(
                int(center_x - x_size / 2), int(center_y - x_size / 2),
                int(center_x + x_size / 2), int(center_y + x_size / 2)
            )
            painter.drawLine(
                int(center_x - x_size / 2), int(center_y + x_size / 2),
                int(center_x + x_size / 2), int(center_y - x_size / 2)
            )
    
    def toggle_state(self, active):
        self.is_active = active
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.chats = {}  # Store chat data
        self.current_chat_id = None
        self.is_sidebar_visible = True
        self.setup_ui()
        self.initialize_first_chat()
        self.setWindowTitle("JARVIS AI")
        self.setMinimumSize(1000, 700)
        self.apply_theme()

    def apply_theme(self):
        """Apply current theme to all components."""
        # Apply to main window
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {Colors.BACKGROUND};
            }}
            QSplitter::handle {{
                background-color: {Colors.DIVIDER};
                width: 1px;
            }}
            QWidget#central_widget {{
                background-color: {Colors.BACKGROUND};
            }}
            QPushButton#toggle_sidebar_btn:hover {{
                background-color: {Colors.SIDEBAR_ITEM_HOVER};
            }}
        """)
        
        # Update font sizes
        app_font = self.font()
        app_font.setPointSize(11)
        self.setFont(app_font)
        
        # Update child widgets
        if hasattr(self, 'sidebar'):
            self.sidebar.update_style()
        
        if hasattr(self, 'chat_widget'):
            self.chat_widget.update_style()
            
        if hasattr(self, 'input_widget'):
            self.input_widget.update_style()

    def setup_ui(self):
        # Create main container
        self.central_widget = QWidget()
        self.central_widget.setObjectName("central_widget")
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create splitter for sidebar and content
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Create sidebar
        self.sidebar = SidebarWidget()
        self.sidebar.new_chat_clicked.connect(self.create_new_chat)
        self.sidebar.chat_selected.connect(self.switch_to_chat)
        self.sidebar.settings_clicked.connect(self.open_settings)
        
        # Create content area with toggle button
        self.content_container = QWidget()
        content_layout = QVBoxLayout(self.content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Header with toggle button
        header = QWidget()
        header.setFixedHeight(50)
        header.setStyleSheet(f"background-color: {Colors.BACKGROUND};")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 10, 0)
        
        # Toggle sidebar button with animated icon
        self.toggle_sidebar_btn = AnimatedToggleButton()
        self.toggle_sidebar_btn.clicked.connect(self.toggle_sidebar)
        header_layout.addWidget(self.toggle_sidebar_btn)
        header_layout.addStretch()
        
        # Chat and input areas
        chat_input_container = QWidget()
        chat_input_layout = QVBoxLayout(chat_input_container)
        chat_input_layout.setContentsMargins(0, 0, 0, 0)
        chat_input_layout.setSpacing(0)
        
        self.chat_widget = ChatWidget()
        self.input_widget = InputWidget()
        
        chat_input_layout.addWidget(self.chat_widget)
        chat_input_layout.addWidget(self.input_widget)
        
        # Add content sections
        content_layout.addWidget(header)
        content_layout.addWidget(chat_input_container)
        
        # Add to splitter
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.content_container)
        
        # Set initial sizes
        self.splitter.setSizes([250, 750])
        self.splitter.setHandleWidth(1)
        self.splitter.setCollapsible(1, False)  # Content area can't be collapsed
        
        # Add splitter to main layout
        main_layout.addWidget(self.splitter)
        
        # Connect signals
        self.input_widget.message_sent.connect(self.add_user_message)
        
        # Set central widget
        self.setCentralWidget(self.central_widget)
        
    def initialize_first_chat(self):
        """Create the initial chat automatically."""
        self.create_new_chat()
    
    def create_new_chat(self):
        """Create a new chat and switch to it."""
        chat_id = str(uuid.uuid4())
        chat_title = f"Chat {len(self.chats) + 1}"
        
        # Store chat data
        self.chats[chat_id] = {
            'title': chat_title,
            'messages': []
        }
        
        # Add to sidebar
        self.sidebar.add_chat(chat_id, chat_title)
        
        # Switch to this chat
        self.switch_to_chat(chat_id)
        
    def switch_to_chat(self, chat_id):
        """Switch to the specified chat."""
        if chat_id in self.chats:
            self.current_chat_id = chat_id
            self.sidebar.set_active_chat(chat_id)
            
            # Clear the chat widget and load the messages
            self.chat_widget.set_chat(chat_id)
            
            # Reload messages
            for msg in self.chats[chat_id]['messages']:
                self.chat_widget.add_message(msg['text'], msg['is_user'])
    
    def toggle_sidebar(self):
        """Show or hide the sidebar with smooth animation."""
        current_sizes = self.splitter.sizes()
        
        # Toggle button animation
        self.toggle_sidebar_btn.toggle_state(not self.is_sidebar_visible)
        
        if self.is_sidebar_visible:
            # Hide sidebar with animation
            start_width = current_sizes[0]
            animation = self._create_sidebar_animation(start_width, 0)
            animation.valueChanged.connect(lambda w: self.splitter.setSizes([w, current_sizes[1] + start_width - w]))
        else:
            # Show sidebar with animation
            animation = self._create_sidebar_animation(0, 250)
            animation.valueChanged.connect(lambda w: self.splitter.setSizes([w, current_sizes[1] - w]))
        animation.start()
        
        # Toggle state
        self.is_sidebar_visible = not self.is_sidebar_visible
    
    def _create_sidebar_animation(self, start_value, end_value):
        """Create a sidebar animation between the given values."""
        animation = QPropertyAnimation(self, b"sidebar_width")
        animation.setDuration(300)
        animation.setStartValue(start_value)
        animation.setEndValue(end_value)
        animation.setEasingCurve(QEasingCurve.InOutCubic)
        return animation
    
    def get_sidebar_width(self):
        return self.splitter.sizes()[0]
        
    def set_sidebar_width(self, width):
        current_sizes = self.splitter.sizes()
        self.splitter.setSizes([width, current_sizes[1]])
    
    # Define property for animation
    sidebar_width = property(get_sidebar_width, set_sidebar_width)
    
    def open_settings(self):
        """Open the settings dialog."""
        dialog = SettingsDialog(self)
        dialog.settings_updated.connect(self.apply_theme)
        dialog.exec_()
    
    def add_user_message(self, text):
        """Add a user message to the current chat."""
        if not self.current_chat_id:
            self.create_new_chat()
            
        # Add message to UI
        user_msg = self.chat_widget.add_message(text, is_user=True)
        
        # Store message in chat data
        self.chats[self.current_chat_id]['messages'].append({
            'text': text,
            'is_user': True,
            'timestamp': time.time()
        })
        
        # Show typing indicator
        typing_indicator = self.chat_widget.typing_indicator()
        
        # Simulate AI response delay
        QTimer.singleShot(1500, lambda: self.process_ai_response(text, typing_indicator))
    
    def process_ai_response(self, user_text, typing_indicator=None):
        """Process user message and generate AI response."""
        # Remove typing indicator if it exists
        if typing_indicator:
            self.chat_widget.layout.removeWidget(typing_indicator)
            typing_indicator.deleteLater()
        
        # In a real app, this would call the AI model
        # For now, we'll provide some canned responses
        if "hello" in user_text.lower():
            ai_response = "Hello! I'm JARVIS, your personal AI assistant. How can I help you today?"
        elif "your name" in user_text.lower():
            ai_response = "I'm JARVIS, an AI assistant designed to help with various tasks."
        elif "help" in user_text.lower():
            ai_response = "I can provide information, answer questions, have conversations, and assist with various tasks. Just let me know what you need help with!"
        elif "thank" in user_text.lower():
            ai_response = "You're welcome! Let me know if you need anything else."
        else:
            ai_response = "I understand you said: \"" + user_text + "\". How else can I assist you today?"
        
        # Add AI message to UI
        self.add_ai_message(ai_response)
    
    def add_ai_message(self, text):
        """Add an AI message to the current chat."""
        if not self.current_chat_id:
            return
            
        # Add message to UI
        ai_msg = self.chat_widget.add_message(text, is_user=False)
        
        # Store message in chat data
        self.chats[self.current_chat_id]['messages'].append({
            'text': text,
            'is_user': False,
            'timestamp': time.time()
        })
    
    def regenerate_response(self):
        """Regenerate the last AI response."""
        if not self.current_chat_id or not self.chats[self.current_chat_id]['messages']:
            return
            
        if last_user_msg := next(
            (
                msg['text']
                for msg in reversed(self.chats[self.current_chat_id]['messages'])
                if msg['is_user']
            ),
            None,
        ):
            # Remove the last AI message (both from UI and data)
            if self.chats[self.current_chat_id]['messages'][-1]['is_user'] == False:
                # Remove from data
                self.chats[self.current_chat_id]['messages'].pop()
                
                # Remove from UI (last widget in the layout)
                last_widget = self.chat_widget.layout.itemAt(self.chat_widget.layout.count() - 1)
                if last_widget and last_widget.widget():
                    widget = last_widget.widget()
                    self.chat_widget.layout.removeWidget(widget)
                    widget.deleteLater()
            
            # Show typing indicator and process response again
            typing_indicator = self.chat_widget.typing_indicator()
            QTimer.singleShot(1500, lambda: self.process_ai_response(last_user_msg, typing_indicator))
    
    def edit_user_message(self, message_widget, original_text):
        """Edit a user message and regenerate the AI response."""
        from PyQt5.QtWidgets import QInputDialog
        
        new_text, ok = QInputDialog.getText(
            self, 'Edit Message', 'Edit your message:',
            text=original_text
        )
        
        if ok and new_text and new_text != original_text:
            # Update UI
            message_widget.text = new_text
            message_widget.bubble.setText(new_text)
            
            # Find this message and subsequent AI response in the chat history
            if not self.current_chat_id:
                return
                
            # Update in chat data and find position
            position = -1
            for i, msg in enumerate(self.chats[self.current_chat_id]['messages']):
                if msg['is_user'] and msg['text'] == original_text:
                    # Update the message
                    self.chats[self.current_chat_id]['messages'][i]['text'] = new_text
                    position = i
                    break
            
            if position >= 0 and position < len(self.chats[self.current_chat_id]['messages']) - 1:
                # If there's a message after this one (likely an AI response), regenerate it
                next_msg = self.chats[self.current_chat_id]['messages'][position + 1]
                if not next_msg['is_user']:
                    # Remove AI response
                    self.chats[self.current_chat_id]['messages'].pop(position + 1)
                    
                    # Remove from UI
                    # We need to find the widget index - it's position + 1 in the layout
                    # (adding 1 because of potential empty label at the beginning)
                    ai_widget_idx = position + 1
                    if self.chat_widget.empty_label.isVisible():
                        ai_widget_idx += 1
                        
                    if ai_widget_idx < self.chat_widget.layout.count():
                        widget_item = self.chat_widget.layout.itemAt(ai_widget_idx)
                        if widget_item and widget_item.widget():
                            widget = widget_item.widget()
                            self.chat_widget.layout.removeWidget(widget)
                            widget.deleteLater()
                    
                    # Show typing indicator and process response again
                    typing_indicator = self.chat_widget.typing_indicator()
                    QTimer.singleShot(1500, lambda: self.process_ai_response(new_text, typing_indicator))
                    
    def resizeEvent(self, event):
        """Handle window resize event to adjust UI proportions."""
        super().resizeEvent(event)
        if self.is_sidebar_visible:
            # Maintain sidebar proportion when resizing
            total_width = self.width()
            sidebar_ratio = 0.25  # 25% of window width
            sidebar_width = int(total_width * sidebar_ratio)
            content_width = total_width - sidebar_width
            self.splitter.setSizes([sidebar_width, content_width])
                    
    def closeEvent(self, event):
        """Handle application close event."""
        reply = QMessageBox.question(
            self, 'Exit', 'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Cleanup and save any necessary data
            event.accept()
        else:
            event.ignore()