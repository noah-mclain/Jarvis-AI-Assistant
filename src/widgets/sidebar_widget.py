from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                           QScrollArea, QLabel, QFrame, QSizePolicy)
from PySide6.QtCore import Qt, Signal, QSize, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
from PySide6.QtGui import QIcon, QFont, QColor

from styles.colors import Colors
from styles.animations import fade_in, slide_in, scale, bounce_in

class ChatHistoryItem(QFrame):
    clicked = Signal(str)
    
    def __init__(self, chat_id, title, parent=None):
        super().__init__(parent)
        self.chat_id = chat_id
        self.title = title
        self.is_active = False
        self.setup_ui()
        self.is_hovering = False
        
    def setup_ui(self):
        self.setFixedHeight(50)
        self.setMinimumWidth(200)
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        
        self.title_label = QLabel(self.title)
        self.title_label.setFont(QFont("SF Pro Display, Helvetica Neue, Segoe UI", 12))
        layout.addWidget(self.title_label)
        
        self.update_style()
        
    def update_style(self):
        """Update styling based on current theme colors."""
        bg_color = Colors.SIDEBAR_ITEM_ACTIVE if self.is_active else Colors.SIDEBAR_ITEM
        hover_color = Colors.SIDEBAR_ITEM_ACTIVE if self.is_active else Colors.SIDEBAR_ITEM_HOVER
        text_color = "#ffffff" if self.is_active else Colors.TEXT
        font_weight = "600" if self.is_active else "400"
        
        self.title_label.setStyleSheet(f"color: {text_color}; font-weight: {font_weight}; background-color: transparent;")
        
        self.setStyleSheet(f"""
            ChatHistoryItem {{
                background-color: {bg_color};
                border-radius: 10px;
                margin: 3px 0px;
            }}
            ChatHistoryItem:hover {{
                background-color: {hover_color};
                transition: background-color 0.2s ease;
            }}
        """)
        
    def set_active(self, active):
        """Set the active state with a smooth animation."""
        if self.is_active == active:
            return
            
        self.is_active = active
        
        # Create background color animation
        if active:
            # Add a subtle bounce effect when activated
            bounce = bounce_in(self, direction='right', distance=5, duration=350)
            bounce.start()
        
        self.update_style()
        
    def mousePressEvent(self, event):
        """Handle mouse press with animation."""
        # Scale down slightly on click
        scale_anim = scale(self, end_scale=0.97, duration=100)
        scale_anim.start()
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release with animation."""
        # Scale back to normal
        scale_anim = scale(self, end_scale=1.0, duration=150, ease=QEasingCurve.OutQuad)
        scale_anim.start()
        
        # Emit clicked signal only on release
        self.clicked.emit(self.chat_id)
        super().mouseReleaseEvent(event)
        
    def enterEvent(self, event):
        """Handle mouse enter with smooth animation."""
        self.is_hovering = True
        if not self.is_active:
            anim_group = QParallelAnimationGroup(self)
            
            # Slight scale up
            scale_anim = scale(self, end_scale=1.02, duration=150, ease=QEasingCurve.OutQuad)
            
            # Add animations to group
            anim_group.addAnimation(scale_anim)
            anim_group.start()
            
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """Handle mouse leave with smooth animation."""
        self.is_hovering = False
        if not self.is_active:
            anim_group = QParallelAnimationGroup(self)
            
            # Scale back to normal
            scale_anim = scale(self, end_scale=1.0, duration=150, ease=QEasingCurve.InOutQuad)
            
            # Add animations to group
            anim_group.addAnimation(scale_anim)
            anim_group.start()
            
        super().leaveEvent(event)

class SidebarWidget(QWidget):
    chat_selected = Signal(str)
    new_chat_clicked = Signal()
    settings_clicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.chat_items = {}
        self.active_chat_id = None
        self.setup_ui()
        
        # Add entrance animation
        entrance_anim = QParallelAnimationGroup(self)
        fade = fade_in(self, duration=400)
        slide = slide_in(self, direction='left', distance=20, duration=400, ease=QEasingCurve.OutQuint)
        entrance_anim.addAnimation(fade)
        entrance_anim.addAnimation(slide)
        entrance_anim.start()
        
    def setup_ui(self):
        self.setMinimumWidth(250)
        self.setMaximumWidth(300)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(12, 18, 12, 18)
        self.layout.setSpacing(12)
        
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
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        
        self.history_container = QWidget()
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.setContentsMargins(0, 4, 0, 4)
        self.history_layout.setSpacing(6)
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
        
        # Update buttons with softer styling
        button_style = f"""
            QPushButton {{
                background-color: {Colors.BUTTON_BG};
                color: {Colors.BUTTON_TEXT};
                border-radius: 10px;
                padding: 12px;
                text-align: left;
                font-size: 13px;
                font-weight: 500;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }}
            QPushButton:hover {{
                background-color: {Colors.BUTTON_HOVER};
                transform: translateY(-1px);
                transition: all 0.2s ease;
            }}
            QPushButton:pressed {{
                background-color: {Colors.PRIMARY};
                transform: translateY(1px);
                transition: all 0.1s ease;
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
                width: 5px;
                background: rgba(0, 0, 0, 0.05);
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
        
        # Update all chat items
        for chat_item in self.chat_items.values():
            chat_item.update_style()
        
    def add_chat(self, chat_id, title):
        """Add a new chat to the sidebar with animation."""
        chat_item = ChatHistoryItem(chat_id, title)
        chat_item.clicked.connect(self.on_chat_clicked)
        
        self.history_layout.addWidget(chat_item)
        self.chat_items[chat_id] = chat_item
        chat_item.setGraphicsEffect(None)  # Clear any existing effects
        
        # Add combination of animations
        anim_group = QParallelAnimationGroup(chat_item)
        
        # Slide in from left
        slide_anim = slide_in(chat_item, direction='left', distance=20, duration=350, ease=QEasingCurve.OutQuint)
        
        # Fade in
        fade_anim = fade_in(chat_item, duration=350)
        
        # Add to group and start
        anim_group.addAnimation(slide_anim)
        anim_group.addAnimation(fade_anim)
        anim_group.start()
        
        return chat_item
        
    def on_chat_clicked(self, chat_id):
        """Handle chat item selection."""
        self.set_active_chat(chat_id)
        self.chat_selected.emit(chat_id)
        
    def set_active_chat(self, chat_id):
        """Set the active chat item with fluid animation."""
        # Deactivate current active chat
        if self.active_chat_id and self.active_chat_id in self.chat_items:
            self.chat_items[self.active_chat_id].set_active(False)
        
        # Activate new chat
        if chat_id in self.chat_items:
            self.chat_items[chat_id].set_active(True)
            self.active_chat_id = chat_id
        
    def remove_chat(self, chat_id):
        """Remove a chat from the sidebar with fade out animation."""
        if chat_id in self.chat_items:
            item = self.chat_items[chat_id]
            
            # Create a fade out animation
            fade_out = QPropertyAnimation(item, "windowOpacity")
            fade_out.setDuration(250)
            fade_out.setStartValue(1.0)
            fade_out.setEndValue(0.0)
            fade_out.setEasingCurve(QEasingCurve.InOutQuad)
            
            # When animation finishes, remove the widget
            def finish_remove():
                self.history_layout.removeWidget(item)
                item.deleteLater()
                del self.chat_items[chat_id]
                
                if self.active_chat_id == chat_id:
                    self.active_chat_id = None
            
            fade_out.finished.connect(finish_remove)
            fade_out.start()
            
    def resizeEvent(self, event):
        """Handle widget resize event with fluid resizing."""
        super().resizeEvent(event)
        
        # Adjust chat items to fit new width
        for item in self.chat_items.values():
            item.setMinimumWidth(self.width() - 24)  # Accounting for layout margins 