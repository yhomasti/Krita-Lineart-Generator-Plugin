from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSpinBox, QDoubleSpinBox, QComboBox, 
                             QGroupBox, QSlider, QCheckBox, QFileDialog,
                             QProgressBar, QMessageBox, QTabWidget, QTextEdit)
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
        
        self.setWindowTitle("LineArt-O-Matic - Production Ready")
        self.setMinimumSize(1000, 700)
        
        self.init_ui()
        self.show_original_preview()
        
    def init_ui(self):
        layout = QHBoxLayout()
        
        # Left side: Settings
        left_panel = self.create_settings_panel()
        layout.addWidget(left_panel, 1)
        
        # Right side: Tabbed Preview
        right_panel = self.create_preview_tabs()
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
        
        cleanup_layout.addWidget(QLabel("Prune Iterations:"))
        self.prune_spin = QSpinBox()
        self.prune_spin.setRange(0, 5)
        self.prune_spin.setValue(2)
        self.prune_spin.valueChanged.connect(self.on_settings_changed)
        cleanup_layout.addWidget(self.prune_spin)
        
        cleanup_layout.addWidget(QLabel("Min Noise Area (pixels):"))
        self.noise_spin = QSpinBox()
        self.noise_spin.setRange(1, 200)
        self.noise_spin.setValue(10)
        self.noise_spin.valueChanged.connect(self.on_settings_changed)
        cleanup_layout.addWidget(self.noise_spin)
        
        cleanup_group.setLayout(cleanup_layout)
        panel.addWidget(cleanup_group)
        
        # Gap Closing Settings
        gap_group = QGroupBox("Intelligent Gap Closing")
        gap_layout = QVBoxLayout()
        
        self.gap_closing_check = QCheckBox("Enable Gap Closing")
        self.gap_closing_check.setChecked(True)
        self.gap_closing_check.stateChanged.connect(self.on_settings_changed)
        gap_layout.addWidget(self.gap_closing_check)
        
        gap_layout.addWidget(QLabel("Max Gap Distance (px):"))
        self.gap_distance_spin = QSpinBox()
        self.gap_distance_spin.setRange(1, 50)
        self.gap_distance_spin.setValue(15)
        self.gap_distance_spin.setToolTip("Maximum pixel distance to connect strokes")
        self.gap_distance_spin.valueChanged.connect(self.on_settings_changed)
        gap_layout.addWidget(self.gap_distance_spin)
        
        gap_layout.addWidget(QLabel("Angle Tolerance (degrees):"))
        self.gap_angle_spin = QSpinBox()
        self.gap_angle_spin.setRange(0, 90)
        self.gap_angle_spin.setValue(30)
        self.gap_angle_spin.setToolTip("Only connect strokes with similar angles")
        self.gap_angle_spin.valueChanged.connect(self.on_settings_changed)
        gap_layout.addWidget(self.gap_angle_spin)
        
        gap_group.setLayout(gap_layout)
        panel.addWidget(gap_group)
        
        # Process Button
        self.process_btn = QPushButton("🎨 Process")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setMinimumHeight(50)
        self.process_btn.setStyleSheet("font-size: 14pt; font-weight: bold;")
        panel.addWidget(self.process_btn)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        panel.addWidget(self.progress_bar)
        
        # Stats
        self.stats_label = QLabel("Click Process to analyze your artwork")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("padding: 10px; background: #f0f0f0; border-radius: 5px;")
        panel.addWidget(self.stats_label)
        
        panel.addStretch()
        
        # Bottom Buttons
        button_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("✓ Apply to Layer")
        self.apply_btn.clicked.connect(self.apply_to_layer)
        self.apply_btn.setEnabled(False)
        button_layout.addWidget(self.apply_btn)
        
        self.save_svg_btn = QPushButton("💾 Save SVG")
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
    
    def create_preview_tabs(self):
        """Create tabbed preview area"""
        self.tab_widget = QTabWidget()
        
        # Original Tab
        self.original_tab = QLabel()
        self.original_tab.setAlignment(Qt.AlignCenter)
        self.original_tab.setStyleSheet("background: white; border: 1px solid #ccc;")
        self.tab_widget.addTab(self.original_tab, "📄 Original")
        
        # Result Tab
        self.result_tab = QLabel()
        self.result_tab.setAlignment(Qt.AlignCenter)
        self.result_tab.setStyleSheet("background: white; border: 1px solid #ccc;")
        self.tab_widget.addTab(self.result_tab, "✨ Result")
        
        # Analysis Tab
        self.analysis_tab = QLabel()
        self.analysis_tab.setAlignment(Qt.AlignCenter)
        self.analysis_tab.setStyleSheet("background: white; border: 1px solid #ccc;")
        self.tab_widget.addTab(self.analysis_tab, "🔍 Gap Analysis")
        
        # Stats Tab
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("background: white; font-family: monospace;")
        self.tab_widget.addTab(self.stats_text, "📊 Details")
        
        return self.tab_widget
    
    def show_original_preview(self):
        """Show original image in preview"""
        pixmap = self.bgr_to_pixmap(self.bgr)
        self.original_tab.setPixmap(pixmap)
    
    def bgr_to_pixmap(self, bgr_img, max_size=700):
        """Convert BGR image to QPixmap for display"""
        h, w = bgr_img.shape[:2]
        rgb = bgr_img[:, :, ::-1].copy()
        
        # Scale to fit
        scale = min(max_size / w, max_size / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        
        import cv2
        rgb_scaled = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        q_img = QImage(rgb_scaled.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)
    
    def get_config(self):
        mode = "adaptive" if self.mode_combo.currentIndex() == 0 else "xdog"
        return {
            "mode": mode,
            "line_width": self.line_width_spin.value(),
            "prune_iters": self.prune_spin.value(),
            "noise_min_area": self.noise_spin.value(),
            "gap_closing": self.gap_closing_check.isChecked(),
            "max_gap": self.gap_distance_spin.value(),
            "angle_threshold": self.gap_angle_spin.value()
        }
    
    def on_settings_changed(self):
        # Could add auto-preview here
        pass
    
    def process_image(self):
        if self.processing:
            return
        
        self.processing = True
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        config = self.get_config()
        
        self.thread = ProcessingThread(self.core, self.bgr, config)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.error.connect(self.on_processing_error)
        self.thread.start()
    
    def on_processing_finished(self, result):
        self.result = result
        self.processing = False
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Update Result tab
        result_pixmap = self.bgr_to_pixmap(result["preview_bgr"])
        self.result_tab.setPixmap(result_pixmap)
        
        # Update Analysis tab if available
        if result.get("analysis_img") is not None:
            analysis_pixmap = self.bgr_to_pixmap(result["analysis_img"])
            self.analysis_tab.setPixmap(analysis_pixmap)
            self.tab_widget.setTabEnabled(2, True)
        else:
            self.analysis_tab.setText("Gap closing disabled\n\nEnable it in settings to see analysis")
            self.tab_widget.setTabEnabled(2, False)
        
        # Update stats
        stats = result.get("stats", {})
        gap_analysis = result.get("gap_analysis", {})
        
        stats_text = "=== Processing Results ===\n\n"
        stats_text += f"Total Strokes: {stats.get('strokes', 0)}\n"
        stats_text += f"SVG Paths: {stats.get('svg_paths', 0)}\n"
        stats_text += f"Pixels Detected: {stats.get('pixels', 0):,}\n\n"
        
        if gap_analysis:
            stats_text += "=== Gap Closing Analysis ===\n\n"
            stats_text += f"Original Strokes: {gap_analysis.get('original_count', 0)}\n"
            stats_text += f"Final Strokes: {gap_analysis.get('final_count', 0)}\n"
            stats_text += f"✓ Gaps Closed: {gap_analysis.get('gaps_closed', 0)}\n"
            stats_text += f"✗ Gaps Rejected: {gap_analysis.get('gaps_rejected', 0)}\n\n"
            
            if gap_analysis.get('successful_connections'):
                stats_text += "Successful Connections:\n"
                for conn in gap_analysis['successful_connections'][:10]:  # Show first 10
                    stats_text += f"  • {conn['distance']:.1f}px gap, {conn['angle']:.1f}° angle\n"
                if len(gap_analysis['successful_connections']) > 10:
                    stats_text += f"  ... and {len(gap_analysis['successful_connections']) - 10} more\n"
        
        self.stats_text.setText(stats_text)
        self.stats_label.setText(
            f"✓ Processed: {stats.get('strokes', 0)} strokes, "
            f"{gap_analysis.get('gaps_closed', 0) if gap_analysis else 0} gaps closed"
        )
        
        # Enable buttons
        self.apply_btn.setEnabled(True)
        self.save_svg_btn.setEnabled(True)
        
        # Switch to result tab
        self.tab_widget.setCurrentIndex(1)
    
    def on_processing_error(self, error_msg):
        self.processing = False
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Processing Error", f"Failed:\n{error_msg}")
    
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
        
        QMessageBox.information(self, "Success", "Layer added!")
    
    def save_svg(self):
        if not self.result:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save SVG", "", "SVG Files (*.svg)")
        
        if path:
            h, w = self.bgr.shape[:2]
            try:
                self.core.save_svg(self.result["svg_paths"], path, width=w, height=h)
                QMessageBox.information(self, "Success", f"SVG saved!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed:\n{e}")