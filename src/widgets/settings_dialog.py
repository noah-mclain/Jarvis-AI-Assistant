from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                            QComboBox, QTabWidget, QWidget, QColorDialog, QFormLayout,
                            QScrollArea, QFrame, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont, QColor, QPainter, QBrush

from styles.colors import Colors
from styles.animations import fade_in

class ColorSwatch(QFrame):
    clicked = pyqtSignal(str)
    
    def __init__(self, color_name, color_value, parent=None):
        super().__init__(parent)
        self.color_name = color_name
        self.color_value = color_value
        self.setup_ui()
        
    def setup_ui(self):
        self.setFixedSize(60, 60)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            ColorSwatch {{
                background-color: {self.color_value};
                border: 2px solid #888888;
                border-radius: 5px;
            }}
            ColorSwatch:hover {{
                border: 2px solid white;
            }}
        """)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(Qt.white if QColor(self.color_value).lightness() < 128 else Qt.black)
        painter.drawText(self.rect(), Qt.AlignCenter, self.color_name)
        
    def mousePressEvent(self, event):
        self.clicked.emit(self.color_name)
        super().mousePressEvent(event)

class SettingsDialog(QDialog):
    settings_updated = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Settings")
        self.setMinimumSize(500, 400)
        self.setWindowIcon(QIcon("styles/svg/settings_icon.svg"))
        self.setModal(True)

        # Apply current theme colors to dialog
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BACKGROUND};
                color: {Colors.TEXT};
            }}
            QLabel {{
                color: {Colors.TEXT};
            }}
            QPushButton {{
                background-color: {Colors.BUTTON_BG};
                color: {Colors.BUTTON_TEXT};
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {Colors.BUTTON_HOVER};
            }}
            QComboBox {{
                background-color: {Colors.INPUT_BG};
                color: {Colors.TEXT};
                border: 1px solid {Colors.DIVIDER};
                border-radius: 5px;
                padding: 5px;
                min-width: 150px;
            }}
            QTabWidget::pane {{
                border: 1px solid {Colors.DIVIDER};
                background-color: {Colors.BACKGROUND};
            }}
            QTabBar::tab {{
                background-color: {Colors.SIDEBAR_ITEM};
                color: {Colors.TEXT};
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 15px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {Colors.SIDEBAR_ITEM_ACTIVE};
            }}
            QTabBar::tab:hover {{
                background-color: {Colors.SIDEBAR_ITEM_HOVER};
            }}
        """)

        main_layout = QVBoxLayout(self)

        # Create tabs
        self.tab_widget = QTabWidget()

        # Theme tab
        self.theme_tab = QWidget()
        theme_layout = QVBoxLayout(self.theme_tab)

        # Theme selector
        theme_form = QFormLayout()
        theme_form.setSpacing(15)

        self.theme_label = QLabel("Select Theme:")
        self.theme_combo = QComboBox()
        for theme_name in Colors.THEMES.keys():
            display_name = " ".join(word.capitalize() for word in theme_name.split("_"))
            self.theme_combo.addItem(display_name, theme_name)

        if current_theme := self.get_current_theme():
            index = self.theme_combo.findData(current_theme)
            if index >= 0:
                self.theme_combo.setCurrentIndex(index)

        theme_form.addRow(self.theme_label, self.theme_combo)
        theme_layout.addLayout(theme_form)

        # Apply button for theme
        theme_button_layout = QHBoxLayout()
        theme_button_layout.addStretch()

        self.apply_theme_btn = QPushButton("Apply Theme")
        self.apply_theme_btn.clicked.connect(self.apply_theme)
        theme_button_layout.addWidget(self.apply_theme_btn)

        theme_layout.addLayout(theme_button_layout)
        theme_layout.addStretch()

        # Custom Colors tab
        self.colors_tab = QWidget()
        colors_layout = QVBoxLayout(self.colors_tab)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        colors_container = QWidget()
        color_grid = QGridLayout(colors_container)
        color_grid.setSpacing(10)

        # Add color swatches
        self.color_swatches = {}
        row, col = 0, 0
        max_cols = 4

        for color_name, color_value in self.get_current_colors().items():
            if hasattr(Colors, color_name):
                swatch = ColorSwatch(color_name, color_value)
                swatch.clicked.connect(self.on_color_clicked)

                self.color_swatches[color_name] = swatch
                color_grid.addWidget(swatch, row, col)

                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1

        scroll_area.setWidget(colors_container)
        colors_layout.addWidget(scroll_area)

        # Add tabs to widget
        self.tab_widget.addTab(self.theme_tab, "Themes")
        self.tab_widget.addTab(self.colors_tab, "Custom Colors")

        main_layout.addWidget(self.tab_widget)

        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.reset_btn = QPushButton("Reset to Default")
        self.reset_btn.clicked.connect(self.reset_to_default)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)

        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.close_btn)

        main_layout.addLayout(button_layout)

        # Apply fade-in animation
        self.animation = fade_in(self)
        self.animation.start()
    
    def get_current_theme(self):
        """Try to determine the current theme by comparing with presets."""
        for theme_name, theme_colors in Colors.THEMES.items():
            matches = sum(bool(hasattr(Colors, color_name)
                                      and getattr(Colors, color_name) == color_value)
                      for color_name, color_value in theme_colors.items())
            # If most colors match, assume it's this theme
            if matches > len(theme_colors) * 0.8:
                return theme_name

        return "dark_blue"  # Default theme
    
    def get_current_colors(self):
        """Get current color values from Colors class."""
        colors = {}
        for attr in dir(Colors):
            if attr.isupper() and not attr.startswith("_") and attr != "THEMES":
                value = getattr(Colors, attr)
                if isinstance(value, str) and (value.startswith("#") or value.startswith("rgba")):
                    colors[attr] = value
        return colors
    
    def on_color_clicked(self, color_name):
        """Open color picker when a swatch is clicked."""
        if hasattr(Colors, color_name):
            current_color = getattr(Colors, color_name)
            color = QColorDialog.getColor(QColor(current_color), self, f"Select {color_name} Color")
            
            if color.isValid():
                # Update the color
                new_color = color.name()
                Colors.customize_color(color_name, new_color)
                
                # Update the swatch
                self.color_swatches[color_name].color_value = new_color
                self.color_swatches[color_name].setStyleSheet(f"""
                    ColorSwatch {{
                        background-color: {new_color};
                        border: 2px solid #888888;
                        border-radius: 5px;
                    }}
                    ColorSwatch:hover {{
                        border: 2px solid white;
                    }}
                """)
                
                # Emit signal
                self.settings_updated.emit()
    
    def apply_theme(self):
        """Apply the selected theme."""
        theme_name = self.theme_combo.currentData()
        if theme_name and Colors.apply_theme(theme_name):
            # Update swatches
            current_colors = self.get_current_colors()
            for color_name, swatch in self.color_swatches.items():
                if color_name in current_colors:
                    swatch.color_value = current_colors[color_name]
                    swatch.setStyleSheet(f"""
                        ColorSwatch {{
                            background-color: {current_colors[color_name]};
                            border: 2px solid #888888;
                            border-radius: 5px;
                        }}
                        ColorSwatch:hover {{
                            border: 2px solid white;
                        }}
                    """)
            
            # Update dialog appearance
            self.setup_ui()
            
            # Emit signal
            self.settings_updated.emit()
    
    def reset_to_default(self):
        """Reset to default theme."""
        Colors.apply_theme("dark_blue")
        
        # Update theme selector
        index = self.theme_combo.findData("dark_blue")
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)
        
        # Update swatches
        current_colors = self.get_current_colors()
        for color_name, swatch in self.color_swatches.items():
            if color_name in current_colors:
                swatch.color_value = current_colors[color_name]
                swatch.setStyleSheet(f"""
                    ColorSwatch {{
                        background-color: {current_colors[color_name]};
                        border: 2px solid #888888;
                        border-radius: 5px;
                    }}
                    ColorSwatch:hover {{
                        border: 2px solid white;
                    }}
                """)
        
        # Emit signal
        self.settings_updated.emit() 