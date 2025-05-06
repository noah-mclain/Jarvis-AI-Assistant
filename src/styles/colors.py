class Colors:
    # Default theme (Soft Dark)
    PRIMARY = "#7aa2d9"      # Softer Blue
    SECONDARY = "#94a3db"    # Softer Indigo
    BACKGROUND = "#1c2131"   # Soft Dark Blue-Gray
    TEXT = "#e9edf5"         # Soft Off-White
    ERROR = "#e8a4a4"        # Softer Red
    USER_BUBBLE = "#2e3749"  # Soft Dark Gray with slight blue tint
    AI_BUBBLE = "#242d40"    # Darker Gray with blue tint
    INPUT_BG = "#2e3749"     # Soft Dark Gray with slight blue tint
    SIDEBAR_BG = "#161c2d"   # Dark Blue-Gray
    SIDEBAR_ITEM = "#1c2131" # Soft Dark Blue-Gray
    SIDEBAR_ITEM_HOVER = "#394254"  # Medium Gray-Blue
    SIDEBAR_ITEM_ACTIVE = "#7aa2d9" # Softer Blue
    BUTTON_BG = "#7aa2d9"    # Softer Blue
    BUTTON_HOVER = "#8eaede" # Lighter Soft Blue
    BUTTON_TEXT = "#ffffff"  # White
    DIVIDER = "#323b4d"      # Softer divider color
    SCROLLBAR_BG = "rgba(0, 0, 0, 0.05)"
    SCROLLBAR_HANDLE = "rgba(255, 255, 255, 0.15)"
    
    # Theme presets
    THEMES = {
        "dark": {
            "PRIMARY": "#7aa2d9",      # Softer Blue
            "SECONDARY": "#94a3db",    # Softer Indigo
            "BACKGROUND": "#1c2131",   # Soft Dark Blue-Gray
            "TEXT": "#e9edf5",         # Soft Off-White
            "ERROR": "#e8a4a4",        # Softer Red
            "USER_BUBBLE": "#2e3749",  # Soft Dark Gray with slight blue tint
            "AI_BUBBLE": "#242d40",    # Darker Gray with blue tint
            "INPUT_BG": "#2e3749",     # Soft Dark Gray with slight blue tint
            "SIDEBAR_BG": "#161c2d",   # Dark Blue-Gray
            "SIDEBAR_ITEM": "#1c2131", # Soft Dark Blue-Gray
            "SIDEBAR_ITEM_HOVER": "#394254",  # Medium Gray-Blue
            "SIDEBAR_ITEM_ACTIVE": "#7aa2d9", # Softer Blue
            "BUTTON_BG": "#7aa2d9",    # Softer Blue
            "BUTTON_HOVER": "#8eaede", # Lighter Soft Blue
            "BUTTON_TEXT": "#ffffff",  # White
            "DIVIDER": "#323b4d",      # Softer divider color
            "SCROLLBAR_BG": "rgba(0, 0, 0, 0.05)",
            "SCROLLBAR_HANDLE": "rgba(255, 255, 255, 0.15)"
        },
        "dark_purple": {
            "PRIMARY": "#b9aad9",      # Softer Lavender
            "SECONDARY": "#c7b2e3",    # Softer Purple
            "BACKGROUND": "#1c2131",   # Soft Dark Blue-Gray
            "TEXT": "#e9edf5",         # Soft Off-White
            "ERROR": "#e8a4a4",        # Softer Red
            "USER_BUBBLE": "#2e3749",  # Soft Dark Gray with slight blue tint
            "AI_BUBBLE": "#242d40",    # Darker Gray with blue tint
            "INPUT_BG": "#2e3749",     # Soft Dark Gray with slight blue tint
            "SIDEBAR_BG": "#161c2d",   # Dark Blue-Gray
            "SIDEBAR_ITEM": "#1c2131", # Soft Dark Blue-Gray
            "SIDEBAR_ITEM_HOVER": "#394254",  # Medium Gray-Blue
            "SIDEBAR_ITEM_ACTIVE": "#b9aad9", # Softer Lavender
            "BUTTON_BG": "#b9aad9",    # Softer Lavender
            "BUTTON_HOVER": "#c7b2e3", # Softer Purple
            "BUTTON_TEXT": "#ffffff",  # White
            "DIVIDER": "#323b4d",      # Softer divider color
            "SCROLLBAR_BG": "rgba(0, 0, 0, 0.05)",
            "SCROLLBAR_HANDLE": "rgba(255, 255, 255, 0.15)"
        },
        "dark_green": {
            "PRIMARY": "#a4d9c2",      # Softer Mint
            "SECONDARY": "#8ecfbe",    # Softer Seafoam
            "BACKGROUND": "#1c2131",   # Soft Dark Blue-Gray
            "TEXT": "#e9edf5",         # Soft Off-White
            "ERROR": "#e8a4a4",        # Softer Red
            "USER_BUBBLE": "#2e3749",  # Soft Dark Gray with slight blue tint
            "AI_BUBBLE": "#242d40",    # Darker Gray with blue tint
            "INPUT_BG": "#2e3749",     # Soft Dark Gray with slight blue tint
            "SIDEBAR_BG": "#161c2d",   # Dark Blue-Gray
            "SIDEBAR_ITEM": "#1c2131", # Soft Dark Blue-Gray
            "SIDEBAR_ITEM_HOVER": "#394254",  # Medium Gray-Blue
            "SIDEBAR_ITEM_ACTIVE": "#a4d9c2", # Softer Mint
            "BUTTON_BG": "#a4d9c2",    # Softer Mint
            "BUTTON_HOVER": "#8ecfbe", # Softer Seafoam
            "BUTTON_TEXT": "#ffffff",  # White
            "DIVIDER": "#323b4d",      # Softer divider color
            "SCROLLBAR_BG": "rgba(0, 0, 0, 0.05)",
            "SCROLLBAR_HANDLE": "rgba(255, 255, 255, 0.15)"
        },
        "light": {
            "PRIMARY": "#7aa5db",      # Softer Blue
            "SECONDARY": "#a0abe0",    # Softer Periwinkle
            "BACKGROUND": "#f7f9fc",   # Very Light Gray-Blue
            "TEXT": "#2d3748",         # Soft Dark Gray
            "ERROR": "#e8a4a4",        # Softer Red
            "USER_BUBBLE": "#e7ebf3",  # Pale Blue-Gray
            "AI_BUBBLE": "#f2f5fb",    # Even Paler Blue-Gray
            "INPUT_BG": "#ffffff",     # White
            "SIDEBAR_BG": "#eff3f9",   # Pale Gray-Blue
            "SIDEBAR_ITEM": "#f7f9fc", # Very Light Gray-Blue
            "SIDEBAR_ITEM_HOVER": "#dfe6f2",  # Slightly Darker Gray-Blue
            "SIDEBAR_ITEM_ACTIVE": "#7aa5db", # Softer Blue
            "BUTTON_BG": "#7aa5db",    # Softer Blue
            "BUTTON_HOVER": "#93b7eb", # Lighter Soft Blue
            "BUTTON_TEXT": "#ffffff",  # White
            "DIVIDER": "#e7ebf3",      # Pale Blue-Gray
            "SCROLLBAR_BG": "rgba(0, 0, 0, 0.03)",
            "SCROLLBAR_HANDLE": "rgba(0, 0, 0, 0.12)"
        },
        "light_purple": {
            "PRIMARY": "#b9aad9",      # Softer Lavender
            "SECONDARY": "#c7b2e3",    # Softer Purple
            "BACKGROUND": "#f7f9fc",   # Very Light Gray-Blue
            "TEXT": "#2d3748",         # Soft Dark Gray
            "ERROR": "#e8a4a4",        # Softer Red
            "USER_BUBBLE": "#e7ebf3",  # Pale Blue-Gray
            "AI_BUBBLE": "#f2f5fb",    # Even Paler Blue-Gray
            "INPUT_BG": "#ffffff",     # White
            "SIDEBAR_BG": "#eff3f9",   # Pale Gray-Blue
            "SIDEBAR_ITEM": "#f7f9fc", # Very Light Gray-Blue
            "SIDEBAR_ITEM_HOVER": "#dfe6f2",  # Slightly Darker Gray-Blue
            "SIDEBAR_ITEM_ACTIVE": "#b9aad9", # Softer Lavender
            "BUTTON_BG": "#b9aad9",    # Softer Lavender
            "BUTTON_HOVER": "#c7b2e3", # Softer Purple
            "BUTTON_TEXT": "#ffffff",  # White
            "DIVIDER": "#e7ebf3",      # Pale Blue-Gray
            "SCROLLBAR_BG": "rgba(0, 0, 0, 0.03)",
            "SCROLLBAR_HANDLE": "rgba(0, 0, 0, 0.12)"
        },
        "light_green": {
            "PRIMARY": "#a4d9c2",      # Softer Mint
            "SECONDARY": "#8ecfbe",    # Softer Seafoam
            "BACKGROUND": "#f7f9fc",   # Very Light Gray-Blue
            "TEXT": "#2d3748",         # Soft Dark Gray
            "ERROR": "#e8a4a4",        # Softer Red
            "USER_BUBBLE": "#e7ebf3",  # Pale Blue-Gray
            "AI_BUBBLE": "#f2f5fb",    # Even Paler Blue-Gray
            "INPUT_BG": "#ffffff",     # White
            "SIDEBAR_BG": "#eff3f9",   # Pale Gray-Blue
            "SIDEBAR_ITEM": "#f7f9fc", # Very Light Gray-Blue
            "SIDEBAR_ITEM_HOVER": "#dfe6f2",  # Slightly Darker Gray-Blue
            "SIDEBAR_ITEM_ACTIVE": "#a4d9c2", # Softer Mint
            "BUTTON_BG": "#a4d9c2",    # Softer Mint
            "BUTTON_HOVER": "#8ecfbe", # Softer Seafoam
            "BUTTON_TEXT": "#ffffff",  # White
            "DIVIDER": "#e7ebf3",      # Pale Blue-Gray
            "SCROLLBAR_BG": "rgba(0, 0, 0, 0.03)",
            "SCROLLBAR_HANDLE": "rgba(0, 0, 0, 0.12)"
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

    @classmethod
    def is_dark_mode(cls):
        """Check if the current theme is dark mode."""
        return cls.BACKGROUND.startswith("#1") or cls.BACKGROUND.startswith("#0")