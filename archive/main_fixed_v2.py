import sys
import os
from pathlib import Path

# Apply our comprehensive Qt plugin path fix
from qt_plugin_fix import create_qapp

if __name__ == "__main__":
    # Create QApplication with the fixed paths
    app = create_qapp(sys.argv)
    app.setStyle("Fusion")
    
    # Load style sheet with error handling
    try:
        with open("src/styles/theme.qss", "r") as f:
            style = f.read()
            app.setStyleSheet(style)
    except FileNotFoundError:
        print("Warning: theme.qss file not found. Using default styles.")

    # Import MainWindow after QApplication is created with proper paths
    from src.main_window import MainWindow
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec()) 