import sys
from PyQt5.QtWidgets import QApplication
from src.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Load style sheet
    with open("styles/theme.qss", "r") as f:
        style = f.read()
        app.setStyleSheet(style)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())