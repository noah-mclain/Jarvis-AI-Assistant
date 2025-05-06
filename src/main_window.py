from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                          QSplitter, QSizePolicy, QPushButton, QMessageBox, 
                          QMenu)
from PySide6.QtCore import Qt, QSize, QTimer, Signal, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
from PySide6.QtGui import QIcon, QFont, QTransform, QPainter, QPen, QAction, QActionGroup
from widgets.chat_widget import ChatWidget
from widgets.input_widget import InputWidget
from widgets.sidebar_widget import SidebarWidget
from widgets.settings_dialog import SettingsDialog
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
        self.current_theme = "dark"  # Default theme
        self.setup_ui()
        self.initialize_first_chat()
        self.setWindowTitle("JARVIS AI")
        self.setMinimumSize(1000, 700)
        self.apply_theme(self.current_theme)
        
        # Setup animations
        self.setup_animations()

    def setup_animations(self):
        """Setup animations for a more fluid UI experience."""
        # Animation timing
        self.animation_duration = 400
        self.animation_easing = QEasingCurve.OutCubic
        
        # Animations will be created when needed
        self.sidebar_animation = None
        self.content_animation = None

    def apply_theme(self, theme_name=None):
        """Apply theme to all components."""
        if theme_name:
            Colors.apply_theme(theme_name)
            self.current_theme = theme_name
        
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
        print("Setting up UI...")
        
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
        print("Adding sidebar...")
        self.splitter.addWidget(self.sidebar)
        
        # Create content area
        self.content_container = QWidget()
        content_layout = QVBoxLayout(self.content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        self.chat_widget = ChatWidget()
        self.input_widget = InputWidget()
        
        # Add components to content layout
        print("Adding chat widget...")
        content_layout.addWidget(self.chat_widget)
        print("Adding input widget...")
        content_layout.addWidget(self.input_widget)
        
        # Add content area to splitter
        self.splitter.addWidget(self.content_container)
        
        # Set initial sizes
        self.splitter.setSizes([250, 750])  # Adjust these values as needed
        self.setCentralWidget(self.central_widget)
        
        print("UI setup complete.")

    def initialize_first_chat(self):
        """Create the initial chat automatically."""
        self.create_new_chat()
    
    def create_new_chat(self):
        """Create a new chat with a unique ID."""
        chat_id = str(uuid.uuid4())
        chat_title = f"New Chat {len(self.chats) + 1}"
        
        # Add to chat data
        self.chats[chat_id] = {
            'title': chat_title,
            'messages': []
        }
        
        # Add to sidebar
        chat_item = self.sidebar.add_chat(chat_id, chat_title)
        
        # Switch to the new chat
        self.switch_to_chat(chat_id)
        
        return chat_id
        
    def switch_to_chat(self, chat_id):
        """Switch to a different chat by ID."""
        if chat_id in self.chats:
            # Update current chat ID
            self.current_chat_id = chat_id
            
            # Update sidebar selection
            self.sidebar.set_active_chat(chat_id)
            
            # Update chat widget
            self.chat_widget.set_chat(chat_id)
            
            # Load messages
            self._load_chat_messages(chat_id)
    
    def _load_chat_messages(self, chat_id):
        """Load all messages for a chat."""
        if chat_id in self.chats:
            messages = self.chats[chat_id]['messages']
            
            # Clear and reload all messages
            for msg in messages:
                message_widget = self.chat_widget.add_message(
                    msg['text'], 
                    msg['is_user']
                )
    
    def toggle_sidebar(self):
        """Toggle the sidebar visibility with fluid animation."""
        sidebar_width = self.sidebar.width()
        sidebar_visible = sidebar_width > 50
        
        # Toggle button state
        self.toggle_sidebar_btn.toggle_state(not sidebar_visible)
        
        if sidebar_visible:
            # Hide sidebar with animation
            self.splitter.setSizes([0, self.width()])  # Collapse the sidebar
        else:
            # Show sidebar with animation
            self.splitter.setSizes([250, self.width() - 250])  # Expand the sidebar
    
    def open_settings(self):
        """Open settings dialog."""
        dialog = SettingsDialog(self.current_theme, self)
        dialog.theme_changed.connect(self.apply_theme)
        dialog.exec()
    
    def show_theme_menu(self):
        """Show the theme selection menu."""
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {Colors.SIDEBAR_BG};
                color: {Colors.TEXT};
                border: 1px solid {Colors.DIVIDER};
                border-radius: 10px;
                padding: 8px 0px;
            }}
            QMenu::item {{
                padding: 8px 24px;
                border-radius: 4px;
                margin: 2px 4px;
            }}
            QMenu::item:selected {{
                background-color: {Colors.SIDEBAR_ITEM_HOVER};
            }}
        """)
        
        # Light themes
        light_menu = QMenu("Light Themes", menu)
        light_menu.setStyleSheet(menu.styleSheet())
        
        light_action = QAction("Light Blue", self)
        light_action.triggered.connect(lambda: self.apply_theme("light"))
        light_menu.addAction(light_action)
        
        light_purple_action = QAction("Light Purple", self)
        light_purple_action.triggered.connect(lambda: self.apply_theme("light_purple"))
        light_menu.addAction(light_purple_action)
        
        light_green_action = QAction("Light Green", self)
        light_green_action.triggered.connect(lambda: self.apply_theme("light_green"))
        light_menu.addAction(light_green_action)
        
        # Dark themes
        dark_menu = QMenu("Dark Themes", menu)
        dark_menu.setStyleSheet(menu.styleSheet())
        
        dark_action = QAction("Dark Blue", self)
        dark_action.triggered.connect(lambda: self.apply_theme("dark"))
        dark_menu.addAction(dark_action)
        
        dark_purple_action = QAction("Dark Purple", self)
        dark_purple_action.triggered.connect(lambda: self.apply_theme("dark_purple"))
        dark_menu.addAction(dark_purple_action)
        
        dark_green_action = QAction("Dark Green", self)
        dark_green_action.triggered.connect(lambda: self.apply_theme("dark_green"))
        dark_menu.addAction(dark_green_action)
        
        # Add submenus
        menu.addMenu(light_menu)
        menu.addMenu(dark_menu)
        
        # Show the menu below the button
        pos = self.theme_btn.mapToGlobal(self.theme_btn.rect().bottomLeft())
        menu.exec_(pos)
        
    def set_code_generation_mode(self):
        """Set code generation mode."""
        # Update UI state for code generation
        self.chat_widget.scroll_to_bottom()
    
    def set_story_generation_mode(self):
        """Set story generation mode."""
        # Update UI state for story generation
        self.chat_widget.scroll_to_bottom()
    
    def set_image_generation_mode(self):
        """Set image generation mode."""
        # Update UI state for image generation
        self.chat_widget.scroll_to_bottom()
    
    def add_user_message(self, text):
        """Add a user message to the current chat."""
        if self.current_chat_id:
            # Add to UI
            message_widget = self.chat_widget.add_message(text, True)
            
            # Add to data model
            self.chats[self.current_chat_id]['messages'].append({
                'text': text,
                'is_user': True,
                'timestamp': time.time()
            })
            
            # Process response (simulating AI response with typing indicator)
            typing_indicator = self.chat_widget.typing_indicator()
            
            # Simulate processing with a delay
            QTimer.singleShot(1500, lambda: self.process_ai_response(text, typing_indicator))
            
            # Scroll to bottom to show the typing indicator
            self.chat_widget.scroll_to_bottom()
    
    def process_ai_response(self, user_text, typing_indicator=None):
        """Process the user input and generate a response."""
        if not self.current_chat_id:
            return
            
        # Simulate AI processing with a realistic delay
        # In a real application, this would call your AI model
        def generate_response(text):
            # This is where you'd integrate with your AI model
            # For now, return a simple response based on the input
            
            if "hello" in text.lower() or "hi" in text.lower():
                return "Hello! How can I assist you today?"
            elif "help" in text.lower():
                return "I'm here to help. Ask me anything!"
            elif "code" in text.lower() or "program" in text.lower():
                return "I can help you with programming. What specific code do you need assistance with?"
            elif "story" in text.lower():
                return "I'd be happy to tell you a story or help you create one. What kind of story would you like?"
            elif "image" in text.lower() or "picture" in text.lower():
                return "I can generate images based on text descriptions. What would you like to see?"
            else:
                return "I understand you're interested in discussing this topic. Would you like me to provide more information or explore a specific aspect?"
        
        response = generate_response(user_text)
        
        # Remove typing indicator if present
        if typing_indicator:
            typing_indicator.setParent(None)
            typing_indicator.deleteLater()
        
        # Add AI response with slight delay for more natural feel
        QTimer.singleShot(300, lambda: self.add_ai_message(response))
    
    def add_ai_message(self, text):
        """Add an AI message to the current chat."""
        if self.current_chat_id:
            # Add to UI
            message_widget = self.chat_widget.add_message(text, False)
            
            # Add to data model
            self.chats[self.current_chat_id]['messages'].append({
                'text': text,
                'is_user': False,
                'timestamp': time.time()
            })
            
            # Scroll to show the new message
            self.chat_widget.scroll_to_bottom()
    
    def regenerate_response(self):
        """Regenerate the last AI response."""
        if not self.current_chat_id:
            return
            
        chat_messages = self.chats[self.current_chat_id]['messages']
        
        # Find the last user message before the AI response
        last_user_message = None
        for msg in reversed(chat_messages):
            if msg['is_user']:
                last_user_message = msg['text']
                break
                
        if last_user_message:
            # Remove the last AI message from the data model
            if not chat_messages[-1]['is_user']:
                chat_messages.pop()
                
                # Clear and reload all messages
                self.chat_widget.clear_messages()
                self._load_chat_messages(self.current_chat_id)
                
                # Process the last user message again
                typing_indicator = self.chat_widget.typing_indicator()
                self.process_ai_response(last_user_message, typing_indicator)
    
    def edit_user_message(self, message_widget, original_text):
        """Allow editing of a user message with animation."""
        if not self.current_chat_id:
            return
            
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox
        
        # Create an edit dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Message")
        dialog.setMinimumSize(500, 200)
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BACKGROUND};
                border-radius: 10px;
            }}
            QTextEdit {{
                background-color: {Colors.INPUT_BG};
                color: {Colors.TEXT};
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
            }}
            QPushButton {{
                background-color: {Colors.BUTTON_BG};
                color: {Colors.BUTTON_TEXT};
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {Colors.BUTTON_HOVER};
            }}
        """)
        
        # Layout
        layout = QVBoxLayout(dialog)
        
        # Text edit for editing the message
        edit_field = QTextEdit(dialog)
        edit_field.setPlainText(original_text)
        edit_field.setFocus()
        layout.addWidget(edit_field)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Show dialog and get result
        if dialog.exec() == QDialog.Accepted:
            new_text = edit_field.toPlainText().strip()
            if new_text and new_text != original_text:
                # Update in data model
                messages = self.chats[self.current_chat_id]['messages']
                
                # Find and update the message
                for msg in messages:
                    if msg['is_user'] and msg['text'] == original_text:
                        msg['text'] = new_text
                        break
                
                # Clear and reload messages to reflect changes
                self.chat_widget.clear_messages()
                self._load_chat_messages(self.current_chat_id)
    
    def resizeEvent(self, event):
        """Handle window resize event for fluid UI scaling."""
        super().resizeEvent(event)
        
        # If sidebar is hidden, ensure it stays hidden
        if not self.is_sidebar_visible:
            # Reset the splitter sizes to keep sidebar collapsed
            self.splitter.setSizes([0, self.width()])
        
        # Adjust any responsive UI elements
        self.chat_widget.scroll_to_bottom()
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Could save chat history to file here
        
        # Ask for confirmation
        reply = QMessageBox.question(
            self, 'Exit Confirmation',
            'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()