import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QFileDialog, QHBoxLayout, QVBoxLayout, QWidget,
    QSpinBox, QRadioButton, QCheckBox, QProgressBar,
    QMessageBox, QGroupBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from seam_carving_core import SeamCarverCore

class Worker(QThread):
    progress = pyqtSignal(int)
    update = pyqtSignal(object)
    finished = pyqtSignal(object)

    def __init__(self, image, n, mode):
        super().__init__()
        self.image = image
        self.n = n
        self.mode = mode

    def run(self):
        core = SeamCarverCore(self.image)
        for i in range(1, self.n + 1):
            preview, result = core.step(self.mode)
            self.update.emit(preview)
            self.progress.emit(i)
        self.finished.emit(result)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Seam Carving - Multi-scale Enabled")
        self.resize(1200, 800)
        self.image = None
        self.result_image = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # پنل کنترل بالایی
        ctrl_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)

        self.spin = QSpinBox()
        self.spin.setRange(1, 1000)
        self.spin.setValue(50)

        # انتخاب حالت
        mode_group = QGroupBox("Mode")
        m_layout = QVBoxLayout(mode_group)
        self.rb_v = QRadioButton("Vertical")
        self.rb_h = QRadioButton("Horizontal")
        self.rb_s = QRadioButton("Smart")
        self.rb_s.setChecked(True)
        for r in (self.rb_v, self.rb_h, self.rb_s): m_layout.addWidget(r)

        # چک‌باکس مولتی‌اسکیل
        self.chk_multi = QCheckBox("Enable Multi-scale (Faster)")
        self.chk_multi.setToolTip("Downscales the image to 50% size before processing.")

        self.btn_run = QPushButton("Run")
        self.btn_run.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_algorithm)

        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.save_result)

        ctrl_layout.addWidget(self.btn_load)
        ctrl_layout.addWidget(QLabel("Seams:"))
        ctrl_layout.addWidget(self.spin)
        ctrl_layout.addWidget(mode_group)
        ctrl_layout.addWidget(self.chk_multi) # اضافه شدن چک باکس به UI
        ctrl_layout.addWidget(self.btn_run)
        ctrl_layout.addWidget(self.btn_save)

        self.progress = QProgressBar()
        
        # بخش نمایش تصاویر
        img_layout = QHBoxLayout()
        self.before = QLabel("Before")
        self.after = QLabel("After (Processing...)")
        for l in (self.before, self.after):
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l.setStyleSheet("border: 1px solid gray; background: #eee;")
            img_layout.addWidget(l)

        main_layout.addLayout(ctrl_layout)
        main_layout.addWidget(self.progress)
        main_layout.addLayout(img_layout)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image")
        if path:
            raw = np.fromfile(path, np.uint8)
            self.image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            self.display_image(self.image, self.before)
            self.result_image = None

    def run_algorithm(self):
        if self.image is None: return
        
        # اعمال Multi-scale در صورت فعال بودن
        processing_img = self.image.copy()
        if self.chk_multi.isChecked():
            processing_img = SeamCarverCore.downscale(processing_img, scale=0.5)
            # نمایش تصویر کوچک شده در بخش After برای اطلاع کاربر
            self.display_image(processing_img, self.after)

        mode = "smart" if self.rb_s.isChecked() else ("horizontal" if self.rb_h.isChecked() else "vertical")
        self.progress.setMaximum(self.spin.value())
        self.btn_run.setEnabled(False)
        
        self.worker = Worker(processing_img, self.spin.value(), mode)
        self.worker.update.connect(lambda img: self.display_image(img, self.after))
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, img):
        self.result_image = img
        self.btn_run.setEnabled(True)
        QMessageBox.information(self, "Finished", "Image resized successfully!")

    def save_result(self):
        if self.result_image is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG (*.png);;JPG (*.jpg)")
        if path:
            ext = ".png" if path.endswith(".png") else ".jpg"
            res, buf = cv2.imencode(ext, self.result_image)
            if res: buf.tofile(path)

    def display_image(self, img, label):
        rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        label.setPixmap(pix)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
