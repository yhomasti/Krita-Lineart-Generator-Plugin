from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSpinBox, QDoubleSpinBox, QComboBox, 
                             QGroupBox, QSlider, QCheckBox, QFileDialog,
                             QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import numpy as np

class ProcessingThread(QThread):
    """Background thread for processing so UI doesn't freeze"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, core, bgr, config):
        super().__init__()
        self.core = core
        self.bgr = bgr
        self.config = config
    
    def run(self):
        try:
            result = self.core.process_numpy(self.bgr, config=self.config)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class LineArtDialog(QDialog):
    def __init__(self, core, bgr, doc, parent=None):
        super().__init__(parent)
        self.core = core
        self.bgr = bgr
        self.doc = doc
        self.result = None
        self.processing = False
        
        self.setWindowTitle("LineArt-O-Matic")
        self.setMinimumSize(800, 600)
        
        self.init_ui()
        self.update_config()
        
    def init_ui(self):
        layout = QHBoxLayout()
        
        # Left side: Settings
        left_panel = self.create_settings_panel()
        layout.addWidget(left_panel, 1)
        
        # Right side: Preview
        right_panel = self.create_preview_panel()
        layout.addWidget(right_panel, 2)
        
        self.setLayout(layout)
    
    def create_settings_panel(self):
        panel = QVBoxLayout()
        
        # Detection Mode
        mode_group = QGroupBox("Detection Mode")
        mode_layout = QVBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Adaptive Threshold", "XDoG Edge Detection"])
        self.mode_combo.currentIndexChanged.connect(self.on_settings_changed)
        mode_layout.addWidget(QLabel("Algorithm:"))
        mode_layout.addWidget(self.mode_combo)
        
        mode_group.setLayout(mode_layout)
        panel.addWidget(mode_group)
        
        # Line Settings
        line_group = QGroupBox("Line Settings")
        line_layout = QVBoxLayout()
        
        # Line Width
        line_layout.addWidget(QLabel("Line Width:"))
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 10.0)
        self.line_width_spin.setSingleStep(0.5)
        self.line_width_spin.setValue(2.0)
        self.line_width_spin.valueChanged.connect(self.on_settings_changed)
        line_layout.addWidget(self.line_width_spin)
        
        line_group.setLayout(line_layout)
        panel.addWidget(line_group)
        
        # Cleanup Settings
        cleanup_group = QGroupBox("Cleanup Settings")
        cleanup_layout = QVBoxLayout()
        
        # Prune Iterations
        cleanup_layout.addWidget(QLabel("Prune Iterations:"))
        self.prune_spin = QSpinBox()
        self.prune_spin.setRange(0, 5)
        self.prune_spin.setValue(2)
        self.prune_spin.valueChanged.connect(self.on_settings_changed)
        cleanup_layout.addWidget(self.prune_spin)
        
        # Noise Removal
        cleanup_layout.addWidget(QLabel("Min Noise Area (pixels):"))
        self.noise_spin = QSpinBox()
        self.noise_spin.setRange(1, 200)
        self.noise_spin.setValue(10)
        self.noise_spin.valueChanged.connect(self.on_settings_changed)
        cleanup_layout.addWidget(self.noise_spin)
        
        cleanup_group.setLayout(cleanup_layout)
        panel.addWidget(cleanup_group)
        
        # Process Button
        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setMinimumHeight(40)
        panel.addWidget(self.process_btn)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        panel.addWidget(self.progress_bar)
        
        # Stats
        self.stats_label = QLabel("No processing yet")
        self.stats_label.setWordWrap(True)
        panel.addWidget(self.stats_label)
        
        panel.addStretch()
        
        # Bottom Buttons
        button_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply to Layer")
        self.apply_btn.clicked.connect(self.apply_to_layer)
        self.apply_btn.setEnabled(False)
        button_layout.addWidget(self.apply_btn)
        
        self.save_svg_btn = QPushButton("Save SVG")
        self.save_svg_btn.clicked.connect(self.save_svg)
        self.save_svg_btn.setEnabled(False)
        button_layout.addWidget(self.save_svg_btn)
        
        panel.addLayout(button_layout)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        panel.addWidget(close_btn)
        
        container = QGroupBox()
        container.setLayout(panel)
        return container
    
    def create_preview_panel(self):
        panel = QVBoxLayout()
        
        panel.addWidget(QLabel("Preview:"))
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 400)
        self.preview_label.setStyleSheet("QLabel { background-color: white; border: 1px solid #ccc; }")
        self.preview_label.setText("Click 'Process' to generate preview")
        
        panel.addWidget(self.preview_label)
        
        container = QGroupBox()
        container.setLayout(panel)
        return container
    
    def get_config(self):
        mode = "adaptive" if self.mode_combo.currentIndex() == 0 else "xdog"
        return {
            "mode": mode,
            "line_width": self.line_width_spin.value(),
            "prune_iters": self.prune_spin.value(),
            "noise_min_area": self.noise_spin.value()
        }
    
    def on_settings_changed(self):
        # Could add auto-preview here if desired
        pass
    
    def update_config(self):
        # Initial preview of original image
        h, w = self.bgr.shape[:2]
        rgb = self.bgr[:, :, ::-1].copy()  # BGR to RGB
        
        # Scale to fit preview
        max_size = 600
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        import cv2
        rgb_scaled = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        q_img = QImage(rgb_scaled.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.preview_label.setPixmap(pixmap)
    
    def process_image(self):
        if self.processing:
            return
        
        self.processing = True
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        config = self.get_config()
        
        # Process in background thread
        self.thread = ProcessingThread(self.core, self.bgr, config)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.error.connect(self.on_processing_error)
        self.thread.start()
    
    def on_processing_finished(self, result):
        self.result = result
        self.processing = False
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Update preview
        preview_bgr = result["preview_bgr"]
        h, w = preview_bgr.shape[:2]
        rgb = preview_bgr[:, :, ::-1].copy()  # BGR to RGB
        
        # Scale to fit preview
        max_size = 600
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        import cv2
        rgb_scaled = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        q_img = QImage(rgb_scaled.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.preview_label.setPixmap(pixmap)
        
        # Update stats
        stats = result.get("stats", {})
        stats_text = f"Strokes: {stats.get('strokes', 0)}\n"
        stats_text += f"SVG Paths: {stats.get('svg_paths', 0)}\n"
        stats_text += f"Pixels: {stats.get('pixels', 0)}"
        self.stats_label.setText(stats_text)
        
        # Enable action buttons
        self.apply_btn.setEnabled(True)
        self.save_svg_btn.setEnabled(True)
    
    def on_processing_error(self, error_msg):
        self.processing = False
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Processing Error", f"Failed to process:\n{error_msg}")
    
    def apply_to_layer(self):
        if not self.result:
            return
        
        from PyQt5.QtCore import QByteArray
        
        preview_bgr = self.result["preview_bgr"]
        h, w, _ = preview_bgr.shape
        r, g, b = preview_bgr[..., 2], preview_bgr[..., 1], preview_bgr[..., 0]
        a = np.full_like(r, 255, np.uint8)
        rgba = np.dstack([r, g, b, a])
        data = QByteArray(bytes(rgba))
        
        layer = self.doc.createNode("LineArt (preview)", "paintLayer")
        layer.setPixelData(data, 0, 0, w, h)
        self.doc.rootNode().addChildNode(layer, None)
        self.doc.refreshProjection()
        
        QMessageBox.information(self, "Success", "Layer added to document!")
    
    def save_svg(self):
        if not self.result:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save SVG", "", "SVG Files (*.svg)")
        
        if path:
            h, w = self.bgr.shape[:2]
            try:
                self.core.save_svg(self.result["svg_paths"], path, width=w, height=h)
                QMessageBox.information(self, "Success", f"SVG saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save SVG:\n{e}")