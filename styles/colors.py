class Colors:
    # Default theme (Dark Blue)
    PRIMARY = "#2ecc71"
    SECONDARY = "#3498db"
    BACKGROUND = "#2c3e50"
    TEXT = "#ecf0f1"
    ERROR = "#e74c3c"
    USER_BUBBLE = "#2980b9"
    AI_BUBBLE = "#27ae60"
    INPUT_BG = "#34495e"
    SIDEBAR_BG = "#1a2530"
    SIDEBAR_ITEM = "#2c3e50"
    SIDEBAR_ITEM_HOVER = "#3d566e"
    SIDEBAR_ITEM_ACTIVE = "#2980b9"
    BUTTON_BG = "#3498db"
    BUTTON_HOVER = "#2980b9"
    BUTTON_TEXT = "#ffffff"
    DIVIDER = "#4a6378"
    SCROLLBAR_BG = "rgba(0, 0, 0, 0.2)"
    SCROLLBAR_HANDLE = "rgba(255, 255, 255, 0.3)"
    
    # Theme presets
    THEMES = {
        "dark_blue": {
            "PRIMARY": "#2ecc71",
            "SECONDARY": "#3498db",
            "BACKGROUND": "#2c3e50",
            "TEXT": "#ecf0f1",
            "ERROR": "#e74c3c",
            "USER_BUBBLE": "#2980b9",
            "AI_BUBBLE": "#27ae60",
            "INPUT_BG": "#34495e",
            "SIDEBAR_BG": "#1a2530",
            "SIDEBAR_ITEM": "#2c3e50",
            "SIDEBAR_ITEM_HOVER": "#3d566e",
            "SIDEBAR_ITEM_ACTIVE": "#2980b9",
            "BUTTON_BG": "#3498db",
            "BUTTON_HOVER": "#2980b9",
            "BUTTON_TEXT": "#ffffff",
            "DIVIDER": "#4a6378",
            "SCROLLBAR_BG": "rgba(0, 0, 0, 0.2)",
            "SCROLLBAR_HANDLE": "rgba(255, 255, 255, 0.3)"
        },
        "dark_purple": {
            "PRIMARY": "#9b59b6",
            "SECONDARY": "#8e44ad",
            "BACKGROUND": "#2c2c3e",
            "TEXT": "#ecf0f1",
            "ERROR": "#e74c3c",
            "USER_BUBBLE": "#8e44ad",
            "AI_BUBBLE": "#9b59b6",
            "INPUT_BG": "#34343e",
            "SIDEBAR_BG": "#1a1a2e",
            "SIDEBAR_ITEM": "#2c2c3e",
            "SIDEBAR_ITEM_HOVER": "#3d3d4f",
            "SIDEBAR_ITEM_ACTIVE": "#8e44ad",
            "BUTTON_BG": "#9b59b6",
            "BUTTON_HOVER": "#8e44ad",
            "BUTTON_TEXT": "#ffffff",
            "DIVIDER": "#4a4a5e",
            "SCROLLBAR_BG": "rgba(0, 0, 0, 0.2)",
            "SCROLLBAR_HANDLE": "rgba(255, 255, 255, 0.3)"
        },
        "light": {
            "PRIMARY": "#27ae60",
            "SECONDARY": "#3498db",
            "BACKGROUND": "#f5f5f5",
            "TEXT": "#333333",
            "ERROR": "#e74c3c",
            "USER_BUBBLE": "#d6eaf8",
            "AI_BUBBLE": "#d5f5e3",
            "INPUT_BG": "#ffffff",
            "SIDEBAR_BG": "#eeeeee",
            "SIDEBAR_ITEM": "#f5f5f5",
            "SIDEBAR_ITEM_HOVER": "#e0e0e0",
            "SIDEBAR_ITEM_ACTIVE": "#3498db",
            "BUTTON_BG": "#3498db",
            "BUTTON_HOVER": "#2980b9",
            "BUTTON_TEXT": "#ffffff",
            "DIVIDER": "#dddddd",
            "SCROLLBAR_BG": "rgba(0, 0, 0, 0.1)",
            "SCROLLBAR_HANDLE": "rgba(0, 0, 0, 0.2)"
        }
    }
    
    @classmethod
    def apply_theme(cls, theme_name):
        """Apply a theme by name from the predefined themes."""
        if theme_name in cls.THEMES:
            for key, value in cls.THEMES[theme_name].items():
                setattr(cls, key, value)
            return True
        return False

    @classmethod
    def customize_color(cls, color_name, color_value):
        """Customize a specific color."""
        if hasattr(cls, color_name):
            setattr(cls, color_name, color_value)
            return True
        return False