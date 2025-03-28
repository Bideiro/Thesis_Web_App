from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
import sys
import time

# Step 4: Create a Worker Class
class Worker(QThread):
    # Custom signal to send data to the main thread
    # what tyope to send 
    update_signal = pyqtSignal(str)

    def run(self):
        for i in range(5):
            time.sleep(1)  # Simulate a time-consuming task
            # sending the shitz
            self.update_signal.emit(f"Processing... {i + 1}")

# Step 5: Create the Main GUI Application
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QThread Crash Course")
        self.setGeometry(100, 100, 300, 200)

        # UI elements
        self.layout = QVBoxLayout()
        self.label = QLabel("Click 'Start' to process in the background")
        self.button = QPushButton("Start")

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

        # Worker thread instance
        # change label shit
        self.worker = Worker()
        self.worker.update_signal.connect(self.update_label)  # Connect signal to the slot

        self.button.clicked.connect(self.start_thread)

    def start_thread(self):
        if not self.worker.isRunning():
            self.worker.start()

    def update_label(self, message):
        self.label.setText(message)

# Step 6: Run the Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
