from krita import Extension, Krita
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtCore import QByteArray
import os, sys, site

ACTION_ID   = "lineart_o_matic_action"
ACTION_TEXT = "LineArt-O-Matic"

class LineArtOMatic(Extension):
    def __init__(self, parent):
        super().__init__(parent)
        self.core = None  # lazy init

    def setup(self):
        pass  # unused on 5.2.x

    def createActions(self, window):
        action = window.createAction(ACTION_ID, ACTION_TEXT, "tools/scripts")
        action.triggered.connect(self.run)

    # ---------- deps + core ----------
    def _ensure_deps_and_core(self):
        # make sure Krita sees user-site (usually already true)
        try: site.addsitedir(site.USER_SITE)
        except Exception: pass
        try:
            import numpy as _np  # noqa
            import cv2  # noqa
        except Exception as e:
            raise RuntimeError(
                "NumPy / OpenCV not found in Krita's Python.\n\n"
                "Install with system Python (3.10 x64):\n"
                "  py -3.10 -m pip install --user numpy opencv-python-headless\n\n"
                f"Details: {e}"
            )
        if self.core is None:
            from .lineartomatic_core import LineArtOMaticCore
            self.core = LineArtOMaticCore()

    # ---------- pixel I/O (no cv2 needed here) ----------
    def _active_layer_to_bgr(self):
        import numpy as np
        app = Krita.instance()
        doc = app.activeDocument()
        if not doc: return None, None
        node = doc.activeNode()
        if not node: return None, None
        w, h = doc.width(), doc.height()
        raw = node.pixelData(0, 0, w, h)  # RGBA8 bytes
        arr = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(h, w, 4)
        bgr = arr[:, :, :3][:, :, ::-1].copy()  # RGBA->BGR
        return bgr, doc

    def _add_layer_from_bgr(self, doc, bgr, name="LineArt (preview)"):
        import numpy as np
        h, w, _ = bgr.shape
        r, g, b = bgr[...,2], bgr[...,1], bgr[...,0]
        a = np.full_like(r, 255, np.uint8)
        rgba = np.dstack([r, g, b, a])
        data = QByteArray(bytes(rgba))
        layer = doc.createNode(name, "paintLayer")
        layer.setPixelData(data, 0, 0, w, h)
        doc.rootNode().addChildNode(layer, None)
        doc.refreshProjection()

    # ---------- action ----------
    def run(self):
        try:
            self._ensure_deps_and_core()
        except Exception as e:
            QMessageBox.critical(None, "LineArt-O-Matic", str(e))
            return

        pack = self._active_layer_to_bgr()
        if pack is None:
            QMessageBox.information(None, "LineArt-O-Matic",
                                    "Open a document and select a raster layer first.")
            return
        bgr, doc = pack

        # Show dialog instead of direct processing
        from .lineartomatic_dialog import LineArtDialog
        
        dialog = LineArtDialog(self.core, bgr, doc)
        dialog.exec_()

        cfg = dict(mode="adaptive", line_width=2, prune_iters=2, noise_min_area=10)

        try:
            result = self.core.process_numpy(bgr, config=cfg)
        except Exception as e:
            QMessageBox.critical(None, "LineArt-O-Matic", f"Processing failed:\n{e}")
            return

        self._add_layer_from_bgr(doc, result["preview_bgr"], "LineArt (preview)")

        if result.get("svg_paths"):
            path, _ = QFileDialog.getSaveFileName(
                None, "Save SVG", "", "SVG Files (*.svg)")
            if path:
                h, w = bgr.shape[:2]
                try:
                    self.core.save_svg(result["svg_paths"], path, width=w, height=h)
                except Exception as e:
                    QMessageBox.critical(None, "LineArt-O-Matic", f"SVG save failed:\n{e}")

Krita.instance().addExtension(LineArtOMatic(Krita.instance()))
