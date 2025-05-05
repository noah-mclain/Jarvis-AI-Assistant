from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                           QScrollArea, QLabel, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont

from styles.colors import Colors
from styles.animations import fade_in, slide_in, scale

class ChatHistoryItem(QFrame):
    clicked = pyqtSignal(str)
    
    def __init__(self, chat_id, title, parent=None):
        super().__init__(parent)
        self.chat_id = chat_id
        self.title = title
        self.is_active = False
        self.setup_ui()
        
    def setup_ui(self):
        self.setFixedHeight(50)
        self.setMinimumWidth(200)
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        self.title_label = QLabel(self.title)
        self.title_label.setFont(QFont("SF Pro Display, Helvetica Neue, Segoe UI", 12))
        layout.addWidget(self.title_label)
        
        self.update_style()
        
    def update_style(self):
        """Update styling based on current theme colors."""
        bg_color = Colors.SIDEBAR_ITEM_ACTIVE if self.is_active else Colors.SIDEBAR_ITEM
        hover_color = Colors.SIDEBAR_ITEM_ACTIVE if self.is_active else Colors.SIDEBAR_ITEM_HOVER
        
        self.setStyleSheet(f"""
            ChatHistoryItem {{
                background-color: {bg_color};
                border-radius: 8px;
                margin: 2px 0px;
            }}
            ChatHistoryItem:hover {{
                background-color: {hover_color};
            }}
            QLabel {{
                color: {Colors.TEXT};
                background-color: transparent;
            }}
        """)
        
    def set_active(self, active):
        self.is_active = active
        self.update_style()
        
    def mousePressEvent(self, event):
        self.clicked.emit(self.chat_id)
        super().mousePressEvent(event)
        
    def enterEvent(self, event):
        if not self.is_active:
            scale_anim = scale(self, end_scale=1.02, duration=150)
            scale_anim.start()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        if not self.is_active:
            scale_anim = scale(self, end_scale=1.0, duration=150)
            scale_anim.start()
        super().leaveEvent(event)

class SidebarWidget(QWidget):
    chat_selected = pyqtSignal(str)
    new_chat_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.chat_items = {}
        self.active_chat_id = None
        self.setup_ui()
        self.animation = fade_in(self)
        self.animation.start()
        
    def setup_ui(self):
        self.setMinimumWidth(250)
        self.setMaximumWidth(300)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 15, 10, 15)
        self.layout.setSpacing(10)
        
        # New Chat Button
        self.new_chat_btn = QPushButton("New Chat")
        self.new_chat_btn.setIcon(QIcon("styles/svg/new_chat_icon.svg"))
        self.new_chat_btn.setIconSize(QSize(18, 18))
        self.new_chat_btn.setCursor(Qt.PointingHandCursor)
        self.new_chat_btn.clicked.connect(self.new_chat_clicked)
        self.layout.addWidget(self.new_chat_btn)
        
        # Chat History Scroll Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.history_container = QWidget()
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_layout.setSpacing(5)
        self.history_layout.setAlignment(Qt.AlignTop)
        
        self.scroll_area.setWidget(self.history_container)
        self.layout.addWidget(self.scroll_area, 1)
        
        # Settings Button
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setIcon(QIcon("styles/svg/settings_icon.svg"))
        self.settings_btn.setIconSize(QSize(18, 18))
        self.settings_btn.setCursor(Qt.PointingHandCursor)
        self.settings_btn.clicked.connect(self.settings_clicked)
        self.layout.addWidget(self.settings_btn)
        
        # Apply initial style
        self.update_style()
        
    def update_style(self):
        """Update styling based on current theme colors."""
        # Update sidebar styling
        self.setStyleSheet(f"""
            SidebarWidget {{
                background-color: {Colors.SIDEBAR_BG};
                border-right: 1px solid {Colors.DIVIDER};
            }}
        """)
        
        # Update buttons
        button_style = f"""
            QPushButton {{
                background-color: {Colors.BUTTON_BG};
                color: {Colors.BUTTON_TEXT};
                border-radius: 8px;
                padding: 10px;
                text-align: left;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {Colors.BUTTON_HOVER};
            }}
        """
        
        self.new_chat_btn.setStyleSheet(button_style)
        self.settings_btn.setStyleSheet(button_style)
        
        # Update scroll area
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
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
        
        # Update all chat items
        for chat_item in self.chat_items.values():
            chat_item.update_style()
        
    def add_chat(self, chat_id, title):
        """Add a new chat to the sidebar."""
        chat_item = ChatHistoryItem(chat_id, title)
        chat_item.clicked.connect(self.on_chat_clicked)
        
        self.history_layout.addWidget(chat_item)
        self.chat_items[chat_id] = chat_item
        
        # Add slide-in animation
        slide_anim = slide_in(chat_item, direction='left', duration=300)
        slide_anim.start()
        
        return chat_item
        
    def on_chat_clicked(self, chat_id):
        """Handle chat item selection."""
        self.set_active_chat(chat_id)
        self.chat_selected.emit(chat_id)
        
    def set_active_chat(self, chat_id):
        """Set the active chat item."""
        # Deactivate current active chat
        if self.active_chat_id and self.active_chat_id in self.chat_items:
            self.chat_items[self.active_chat_id].set_active(False)
        
        # Activate new chat
        if chat_id in self.chat_items:
            self.chat_items[chat_id].set_active(True)
            self.active_chat_id = chat_id
        
    def remove_chat(self, chat_id):
        """Remove a chat from the sidebar."""
        if chat_id in self.chat_items:
            item = self.chat_items[chat_id]
            self.history_layout.removeWidget(item)
            item.deleteLater()
            del self.chat_items[chat_id]
            
            if self.active_chat_id == chat_id:
                self.active_chat_id = None
                
    def resizeEvent(self, event):
        """Handle widget resize event."""
        super().resizeEvent(event)
        # Update layout when resized
        self.update() 