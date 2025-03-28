import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer


class WebcamApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Webcam Display with PyQt and OpenCV")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("Webcam Feed")
        self.layout.addWidget(self.label)

        self.start_button = QPushButton("Start Webcam")
        self.start_button.clicked.connect(self.start_webcam)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Webcam")
        self.stop_button.clicked.connect(self.stop_webcam)
        self.layout.addWidget(self.stop_button)

        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.timer.start(20)

    def stop_webcam(self):
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
        self.label.clear()

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        self.stop_webcam()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamApp()
    window.show()
    sys.exit(app.exec_())
