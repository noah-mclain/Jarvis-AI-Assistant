import sys
from PyQt5.QtWidgets import QApplication, QLabel

app = QApplication(sys.argv)
label = QLabel("Hello, PyQt5!")
label.show()
print("PyQt5 initialized successfully!")
sys.exit(app.exec_()) 