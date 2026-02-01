import numpy as np
import cv2
import math
import xml.etree.ElementTree as ET

class LineArtOMaticCore:
    def __init__(self):
        self.cfg = {
            "line_width": 2,
            "prune_iters": 2,
            "mode": "adaptive",  # or "xdog"
            "noise_min_area": 10
        }

    def update(self, config: dict):
        if config:
            self.cfg.update(config or {})

    # ---------- preprocessing ----------
    def normalize_illumination(self, gray):
        bg = cv2.GaussianBlur(gray, (0, 0), 25)
        norm = cv2.divide(gray, bg, scale=255)
        return np.clip(norm, 0, 255).astype(np.uint8)

    def xdog(self, gray, k=1.6, sigma=0.8, p=20, eps=-0.1, phi=10):
        g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
        g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
        D = g1 - p * g2
        D = D / (np.max(np.abs(D)) + 1e-6)
        E = (D >= eps).astype(np.float32)*1.0 + (D < eps) * (1.0 + np.tanh(phi*(D-eps)))
        out = (1.0 - E) * 255
        return out.astype(np.uint8)

    def binarize(self, gray, mode="adaptive"):
        gray = self.normalize_illumination(gray)
        gray = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
        if mode == "xdog":
            edge = self.xdog(gray)
            _, binimg = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            binimg = cv2.bitwise_not(binimg)
        else:
            binimg = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 5
            )
        # remove tiny specks
        nb, labs, stats, _ = cv2.connectedComponentsWithStats(binimg, connectivity=8)
        out = np.zeros_like(binimg)
        for i in range(1, nb):
            if stats[i, cv2.CC_STAT_AREA] >= self.cfg["noise_min_area"]:
                out[labs == i] = 255
        return out

    # ---------- skeletonization - detect the edges of the drawings and produce a 'skeletonized' image of it----------
    def thinning(self, binary):
        #prefer OpenCV ximgproc if present
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        #the fallback is a Zhang–Suen (minimal)
        return self._zhang_suen(binary)

    def _zhang_suen(self, img):
        I = (img > 0).astype(np.uint8).copy()
        changing1 = changing2 = True
        while changing1 or changing2:
            changing1 = []
            rows, cols = I.shape
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    P = I[i-1:i+2, j-1:j+2]
                    p2,p3,p4,p5,p6,p7,p8,p9 = P[0,1],P[0,2],P[1,2],P[2,2],P[2,1],P[2,0],P[1,0],P[0,0]
                    if I[i,j]==1:
                        nb = p2+p3+p4+p5+p6+p7+p8+p9
                        if 2<=nb<=6:
                            A = int((p2==0 and p3==1)) + int((p3==0 and p4==1)) + \
                                int((p4==0 and p5==1)) + int((p5==0 and p6==1)) + \
                                int((p6==0 and p7==1)) + int((p7==0 and p8==1)) + \
                                int((p8==0 and p9==1)) + int((p9==0 and p2==1))
                            if A==1 and (p2*p4*p6==0) and (p4*p6*p8==0):
                                changing1.append((i,j))
            for i,j in changing1: I[i,j]=0

            changing2 = []
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    P = I[i-1:i+2, j-1:j+2]
                    p2,p3,p4,p5,p6,p7,p8,p9 = P[0,1],P[0,2],P[1,2],P[2,2],P[2,1],P[2,0],P[1,0],P[0,0]
                    if I[i,j]==1:
                        nb = p2+p3+p4+p5+p6+p7+p8+p9
                        if 2<=nb<=6:
                            A = int((p2==0 and p3==1)) + int((p3==0 and p4==1)) + \
                                int((p4==0 and p5==1)) + int((p5==0 and p6==1)) + \
                                int((p6==0 and p7==1)) + int((p7==0 and p8==1)) + \
                                int((p8==0 and p9==1)) + int((p9==0 and p2==1))
                            if A==1 and (p2*p4*p8==0) and (p2*p6*p8==0):
                                changing2.append((i,j))
            for i,j in changing2: I[i,j]=0
        return (I*255).astype(np.uint8)

    def prune_spurs(self, skel, iterations=2):
        s = (skel > 0).astype(np.uint8)
        kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], np.uint8)
        for _ in range(iterations):
            nb = cv2.filter2D(s, -1, kernel, borderType=cv2.BORDER_CONSTANT)
            endpoints = ((nb == 11) & (s == 1))
            s[endpoints] = 0
        return (s*255).astype(np.uint8)

    # ---------- stroke tracing ----------
    def skeleton_to_polylines(self, skel):
        sk = (skel > 0).astype(np.uint8)
        H, W = sk.shape
        visited = np.zeros_like(sk, np.uint8)

        def neighbors8(y, x):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and sk[ny, nx] and not visited[ny, nx]:
                        yield ny, nx

        k = np.ones((3, 3), np.uint8); k[1, 1] = 0
        deg = cv2.filter2D(sk, -1, k, borderType=cv2.BORDER_CONSTANT)

        endpoints = np.argwhere((sk > 0) & (deg <= 2))
        polylines = []

        def walk(y, x):
            path = [(x, y)]
            visited[y, x] = 1
            while True:
                nbrs = list(neighbors8(y, x))
                if not nbrs:
                    break
                if len(nbrs) > 1 and len(path) > 1:
                    px, py = path[-2]
                    vx, vy = (x - px, y - py)
                    def score(ny, nx): return vx*(nx - x) + vy*(ny - y)
                    ny, nx = max(nbrs, key=lambda p: score(p[0], p[1]))
                else:
                    ny, nx = nbrs[0]
                path.append((nx, ny))
                visited[ny, nx] = 1
                y, x = ny, nx
            return path

        for y, x in endpoints:
            if not visited[y, x]:
                poly = walk(y, x)
                if len(poly) > 5:
                    polylines.append(poly)

        loops = np.argwhere((sk > 0) & (deg >= 2))
        for y, x in loops:
            if not visited[y, x]:
                poly = walk(y, x)
                if len(poly) > 10:
                    polylines.append(poly)

        return polylines

    # ---------- vectorization ----------
    def simplify_polyline(self, poly, eps=2.0):
        pts = np.array(poly, dtype=np.float32)
        if len(pts) < 3:
            return [tuple(map(int, p)) for p in pts]
        approx = cv2.approxPolyDP(pts, eps, False)
        return [(int(p[0][0]), int(p[0][1])) for p in approx]

    def polylines_to_svg_paths(self, polylines):
        paths = []
        for poly in polylines:
            if len(poly) < 2:
                continue
            sp = self.simplify_polyline(poly, eps=2.0)
            if len(sp) < 2:
                continue
            d = f"M {sp[0][0]},{sp[0][1]}"
            if len(sp) == 2:
                d += f" L {sp[1][0]},{sp[1][1]}"
            else:
                for i in range(1, len(sp) - 1):
                    cx, cy = sp[i]
                    ex, ey = sp[i + 1]
                    d += f" Q {cx},{cy} {ex},{ey}"
            paths.append(d)
        return paths

    def save_svg(self, path_strings, output_path, width, height):
        svg = ET.Element('svg', {
            'width': str(width), 'height': str(height),
            'viewBox': f"0 0 {width} {height}",
            'xmlns': 'http://www.w3.org/2000/svg'
        })
        bg = ET.SubElement(svg, 'rect', {'width': '100%', 'height': '100%', 'fill': 'white'})
        for d in path_strings:
            ET.SubElement(svg, 'path', {
                'd': d, 'stroke': 'black', 'stroke-width': '1.5',
                'fill': 'none', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'
            })
        tree = ET.ElementTree(svg)
        try:
            ET.indent(tree, space="  ", level=0)
        except Exception:
            pass
        tree.write(output_path, encoding='utf-8', xml_declaration=True)

    # ---------- raster preview ----------
    def render_polylines(self, polylines, w, h, thickness=2):
        img = np.full((h, w, 3), 255, np.uint8)
        for poly in polylines:
            if len(poly) < 2: continue
            pts = np.array(poly, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, (0, 0, 0), thickness=max(1, int(thickness)), lineType=cv2.LINE_AA)
        return img

    # ---------- public entry ----------
    def process_numpy(self, img_bgr, config=None):
        self.update(config)
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        binimg = self.binarize(gray, mode=self.cfg["mode"])
        skel  = self.thinning(binimg)
        skel  = self.prune_spurs(skel, iterations=self.cfg["prune_iters"])

        polylines = self.skeleton_to_polylines(skel)
        svg_paths = self.polylines_to_svg_paths(polylines)

        preview = self.render_polylines(polylines, w, h, thickness=self.cfg["line_width"])

        return {
            "preview_bgr": preview,
            "svg_paths": svg_paths,
            "stats": {
                "strokes": len(polylines),
                "svg_paths": len(svg_paths),
                "pixels": int(np.sum(binimg > 0))
            }
        }
