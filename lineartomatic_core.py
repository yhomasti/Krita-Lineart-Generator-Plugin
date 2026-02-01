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
            "noise_min_area": 10,
            "gap_closing": False,
            "max_gap": 15,
            "angle_threshold": 30
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

    # ---------- skeletonization ----------
    def thinning(self, binary):
        # prefer OpenCV ximgproc if present
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        # fallback is Zhang–Suen
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

    # ---------- gap closing with analysis ----------
    def analyze_and_close_gaps(self, polylines, max_gap=15, angle_threshold=30):
        """
        Intelligently connect nearby stroke endpoints with detailed analysis
        Returns: (merged_polylines, gap_analysis)
        """
        
        def get_endpoint_direction(poly, from_start=True):
            """Get the direction vector at an endpoint"""
            if len(poly) < 2:
                return None
            if from_start:
                p1, p2 = poly[0], poly[1]
            else:
                p1, p2 = poly[-1], poly[-2]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length < 0.001:
                return None
            return (dx/length, dy/length)
        
        def angle_between(v1, v2):
            """Calculate angle between two direction vectors (degrees)"""
            if v1 is None or v2 is None:
                return 180
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            dot = max(-1.0, min(1.0, dot))
            return math.degrees(math.acos(abs(dot)))  # abs for direction similarity
        
        def distance(p1, p2):
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            return math.sqrt(dx*dx + dy*dy)
        
        # Build endpoint index
        endpoints = []
        for i, poly in enumerate(polylines):
            if len(poly) >= 2:
                start_dir = get_endpoint_direction(poly, from_start=True)
                end_dir = get_endpoint_direction(poly, from_start=False)
                endpoints.append({
                    'poly_idx': i,
                    'is_start': True,
                    'point': poly[0],
                    'direction': start_dir
                })
                endpoints.append({
                    'poly_idx': i,
                    'is_start': False,
                    'point': poly[-1],
                    'direction': end_dir
                })
        
        # Find potential connections
        potential_connections = []
        rejected_connections = []
        
        for i, ep1 in enumerate(endpoints):
            for j, ep2 in enumerate(endpoints[i+1:], i+1):
                # Don't connect endpoints from same polyline
                if ep1['poly_idx'] == ep2['poly_idx']:
                    continue
                
                dist = distance(ep1['point'], ep2['point'])
                
                # Calculate angle if both directions exist
                angle = 180
                if ep1['direction'] and ep2['direction']:
                    angle = angle_between(ep1['direction'], ep2['direction'])
                
                connection_data = {
                    'ep1_idx': i,
                    'ep2_idx': j,
                    'distance': dist,
                    'angle': angle,
                    'poly1': ep1['poly_idx'],
                    'poly2': ep2['poly_idx'],
                    'point1': ep1['point'],
                    'point2': ep2['point']
                }
                
                # Categorize: accepted or rejected
                if dist <= max_gap and angle <= angle_threshold:
                    potential_connections.append(connection_data)
                elif dist <= max_gap * 1.5:  # Track near-misses for visualization
                    connection_data['reject_reason'] = (
                        'angle too large' if angle > angle_threshold else 'distance too far'
                    )
                    rejected_connections.append(connection_data)
        
        # Sort by distance (connect closest first)
        potential_connections.sort(key=lambda c: c['distance'])
        
        # Merge polylines
        merged = [list(poly) for poly in polylines]
        used_polys = set()
        successful_connections = []
        
        for conn in potential_connections:
            p1_idx = conn['poly1']
            p2_idx = conn['poly2']
            
            # Skip if either polyline already merged
            if p1_idx in used_polys or p2_idx in used_polys:
                continue
            
            ep1 = endpoints[conn['ep1_idx']]
            ep2 = endpoints[conn['ep2_idx']]
            
            poly1 = merged[p1_idx]
            poly2 = merged[p2_idx]
            
            # Merge based on which endpoints connect
            if not ep1['is_start'] and ep2['is_start']:
                new_poly = poly1 + poly2
            elif ep1['is_start'] and not ep2['is_start']:
                new_poly = poly2 + poly1
            elif not ep1['is_start'] and not ep2['is_start']:
                new_poly = poly1 + poly2[::-1]
            else:
                new_poly = poly1[::-1] + poly2
            
            merged[p1_idx] = new_poly
            merged[p2_idx] = []
            used_polys.add(p2_idx)
            successful_connections.append(conn)
        
        # Filter out empty polylines
        result = [p for p in merged if len(p) > 0]
        
        # Build analysis data
        analysis = {
            'original_count': len(polylines),
            'final_count': len(result),
            'gaps_closed': len(successful_connections),
            'gaps_rejected': len(rejected_connections),
            'successful_connections': successful_connections,
            'rejected_connections': rejected_connections
        }
        
        print(f"Gap Analysis: {len(polylines)} -> {len(result)} strokes, "
              f"{len(successful_connections)} gaps closed, "
              f"{len(rejected_connections)} gaps rejected")
        
        return result, analysis

    def render_gap_analysis(self, w, h, analysis, show_accepted=True, show_rejected=True):
        """
        Render a visual analysis of gap closing decisions
        Green = accepted connections
        Red = rejected connections
        """
        img = np.full((h, w, 3), 255, np.uint8)
        
        # Draw rejected connections in red
        if show_rejected:
            for conn in analysis.get('rejected_connections', []):
                p1 = tuple(map(int, conn['point1']))
                p2 = tuple(map(int, conn['point2']))
                cv2.line(img, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(img, p1, 4, (0, 0, 200), -1)
                cv2.circle(img, p2, 4, (0, 0, 200), -1)
                
                # Add label with reason
                mid = ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)
                reason = conn.get('reject_reason', 'rejected')
                cv2.putText(img, reason, mid, cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, (0, 0, 180), 1, cv2.LINE_AA)
        
        # Draw successful connections in green
        if show_accepted:
            for conn in analysis.get('successful_connections', []):
                p1 = tuple(map(int, conn['point1']))
                p2 = tuple(map(int, conn['point2']))
                cv2.line(img, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(img, p1, 4, (0, 200, 0), -1)
                cv2.circle(img, p2, 4, (0, 200, 0), -1)
                
                # Add distance label
                mid = ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)
                label = f"{conn['distance']:.1f}px"
                cv2.putText(img, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, (0, 180, 0), 1, cv2.LINE_AA)
        
        return img

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
        
        gap_analysis = None
        
        # Gap closing with analysis
        if self.cfg.get("gap_closing", False):
            polylines, gap_analysis = self.analyze_and_close_gaps(
                polylines,
                max_gap=self.cfg.get("max_gap", 15),
                angle_threshold=self.cfg.get("angle_threshold", 30)
            )
        
        svg_paths = self.polylines_to_svg_paths(polylines)
        preview = self.render_polylines(polylines, w, h, thickness=self.cfg["line_width"])
        
        # Generate analysis visualization if gap closing was used
        analysis_img = None
        if gap_analysis:
            analysis_img = self.render_gap_analysis(w, h, gap_analysis)

        return {
            "preview_bgr": preview,
            "svg_paths": svg_paths,
            "gap_analysis": gap_analysis,
            "analysis_img": analysis_img,
            "stats": {
                "strokes": len(polylines),
                "svg_paths": len(svg_paths),
                "pixels": int(np.sum(binimg > 0))
            }
        }