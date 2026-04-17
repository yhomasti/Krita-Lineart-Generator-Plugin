#!/usr/bin/env python3
"""
simple line stabilization approach - no complex contours, just enhance what we detect
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime
import tempfile
import zipfile

# create flask app
app = Flask(__name__, 
           static_folder='static',
           template_folder='templates')

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# create directories
def create_directories():
    directories = [
        app.config['UPLOAD_FOLDER'],
        app.config['OUTPUT_FOLDER'], 
        'static/images',
        'static/css',
        'static/js',
        'templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"created/verified directory: {directory}")

create_directories()

class SimpleLineartGenerator:
    """high-resolution line enhancement with color filtering and noise removal"""
    
    def __init__(self):
        self.config = {
            'pencil_sensitivity': 15,
            'color_variance_threshold': 20,
            'graphite_boost': 12,
            'background_filter': 5,
            'noise_removal_threshold': 50,  # new: minimum size for keeping components
            'output_width': 3840,          # 4K resolution
            'output_height': 2160,         # 4K resolution
            'line_width': 2.0,             # medium thickness by default
            'smoothing_iterations': 3,
            'thinning_iterations': 2,
            'super_smoothing': True,
            'curve_smoothing': True,
            'color_processing_mode': 'whole',  # 'whole' or 'ignore-color'
            'anti_aliasing': True
        }

    def update_config(self, new_config):
        self.config.update(new_config)

    def process_image_simple(self, image_path, output_name):
        """simple 3-step process: detect -> thin -> smooth -> vectorize"""
        try:
            start_time = datetime.now()
            print(f"processing with simple approach: {image_path}")
            
            # step 1: load and detect pencil marks (this works perfectly)
            step_start = datetime.now()
            img = self.load_and_prepare_image(image_path)
            original_display = self.resize_for_display(img)
            print(f"step 1 (load): {(datetime.now() - step_start).total_seconds():.2f}s")
            
            # step 2: detect pencil marks (keep the working detection)
            step_start = datetime.now()
            pencil_mask = self.detect_pencil_marks(img)
            mask_display = self.resize_for_display(cv2.cvtColor(pencil_mask, cv2.COLOR_GRAY2BGR))
            print(f"step 2 (detect): {(datetime.now() - step_start).total_seconds():.2f}s")
            
            # step 2.5: clean mask and remove isolated noise
            step_start = datetime.now()
            cleaned_mask = self.clean_mask(pencil_mask)
            print(f"step 2.5 (clean): {(datetime.now() - step_start).total_seconds():.2f}s")
            
            # step 3: thin the lines to get clean centerlines
            step_start = datetime.now()
            thinned_lines = self.thin_lines(cleaned_mask)
            print(f"step 3 (thin): {(datetime.now() - step_start).total_seconds():.2f}s")
            
            # step 4: smooth the thinned lines
            step_start = datetime.now()
            smoothed_lines = self.smooth_lines(thinned_lines)
            print(f"step 4 (smooth): {(datetime.now() - step_start).total_seconds():.2f}s")
            
            # step 5: create final lineart directly from smoothed lines
            step_start = datetime.now()
            lineart_img = self.create_direct_lineart(smoothed_lines)
            lineart_display = self.resize_for_display(lineart_img)
            print(f"step 5 (lineart): {(datetime.now() - step_start).total_seconds():.2f}s")
            
            # step 6: convert to svg paths by tracing the smoothed skeleton
            step_start = datetime.now()
            svg_paths = self.skeleton_to_svg_paths(smoothed_lines)
            svg_path = self.save_svg(svg_paths, output_name)
            print(f"step 6 (svg): {(datetime.now() - step_start).total_seconds():.2f}s")
            
            # save display images
            display_paths = self.save_display_images(
                original_display, mask_display, lineart_display, output_name
            )
            
            total_time = (datetime.now() - start_time).total_seconds()
            print(f"total processing time: {total_time:.2f}s")
            
            return {
                'success': True,
                'svg_path': svg_path,
                'display_images': display_paths,
                'stats': {
                    'lines_detected': int(np.sum(pencil_mask > 0)),  # convert numpy int64 to python int
                    'svg_paths': len(svg_paths),
                    'resolution': f"{self.config['output_width']}×{self.config['output_height']}",
                    'processing_time': f"{total_time:.2f}s"
                }
            }
            
        except Exception as e:
            print(f"error in simple processing: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def load_and_prepare_image(self, image_path):
        """load and scale image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"could not load image: {image_path}")
        
        original_height, original_width = img.shape[:2]
        target_width = self.config['output_width']
        target_height = self.config['output_height']
        
        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = img_resized
        
        return canvas

    def detect_pencil_marks(self, img):
        """pencil detection with optional color filtering"""
        # analyze background
        h, w = img.shape[:2]
        sample_size = min(100, min(h, w) // 4)
        
        samples = []
        samples.append(img[:sample_size, :sample_size])
        samples.append(img[:sample_size, -sample_size:])
        samples.append(img[-sample_size:, :sample_size])
        samples.append(img[-sample_size:, -sample_size:])
        
        all_samples = np.vstack([sample.reshape(-1, 3) for sample in samples])
        bg_color = np.mean(all_samples, axis=0)
        bg_brightness = np.mean(bg_color)
        
        # convert to different color spaces
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        mask = np.zeros(img_gray.shape, dtype=np.uint8)
        
        if self.config.get('color_processing_mode', 'whole') == 'ignore-color':
            # ignore color mode - only detect grayscale/graphite marks
            print("using ignore color mode - only detecting grayscale marks")
            
            # much more restrictive color detection
            saturation = img_hsv[:, :, 1]
            value = img_hsv[:, :, 2]
            
            # only detect low-saturation (grayscale) marks
            grayscale_mask = saturation < 20  # very low saturation
            brightness_mask = value < bg_brightness - (self.config['graphite_boost'] + 5)
            
            # combine for grayscale-only detection
            final_mask = grayscale_mask & brightness_mask
            
        else:
            # whole page mode - detect everything (original behavior)
            print("using whole page mode - detecting all pencil marks")
            
            # multiple detection methods
            brightness_diff = np.abs(img_gray.astype(float) - bg_brightness)
            brightness_mask = brightness_diff > self.config['pencil_sensitivity']
            
            b, g, r = cv2.split(img)
            color_variance = np.maximum(np.maximum(np.abs(r.astype(int) - g.astype(int)),
                                                  np.abs(g.astype(int) - b.astype(int))),
                                       np.abs(r.astype(int) - b.astype(int)))
            color_mask = color_variance > 10
            
            bg_distance = np.sqrt(np.sum((img.astype(float) - bg_color.astype(float))**2, axis=2))
            distance_mask = bg_distance > self.config['color_variance_threshold']
            
            saturation = img_hsv[:, :, 1]
            value = img_hsv[:, :, 2]
            graphite_mask = (saturation < 35) & (value < bg_brightness - self.config['graphite_boost'])
            
            # combine all detection methods
            combined_mask = brightness_mask | (color_mask & distance_mask) | graphite_mask
            bg_filter_mask = bg_distance > self.config['background_filter']
            final_mask = combined_mask & bg_filter_mask
        
        mask[final_mask] = 255
        
        # very light cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        print(f"detected {np.sum(mask > 0)} pencil pixels")
        return mask

    def thin_lines(self, mask):
        """thin detected lines to get clean centerlines - opencv safe version"""
        print("thinning lines to centerlines...")
        
        # convert to binary
        binary = (mask > 127).astype(np.uint8)
        
        # use simple opencv thinning to avoid compatibility issues
        thinned = binary.copy()
        
        # simple iterative thinning
        kernel = np.array([[0, 1, 0],
                          [1, 1, 1], 
                          [0, 1, 0]], dtype=np.uint8)
        
        for i in range(self.config['thinning_iterations']):
            eroded = cv2.erode(thinned, kernel, iterations=1)
            dilated = cv2.dilate(eroded, kernel, iterations=1)
            thinned = cv2.subtract(thinned, dilated)
            thinned = cv2.add(thinned, eroded)
        
        # ensure we have proper binary values
        thinned = np.where(thinned > 0, 255, 0).astype(np.uint8)
        
        print(f"thinned {np.sum(mask > 0)} pixels to {np.sum(thinned > 0)} skeleton pixels")
        return thinned

    def smooth_lines(self, skeleton):
        """advanced high-resolution line smoothing"""
        print("applying advanced high-resolution smoothing...")
        
        smoothed = skeleton.copy()
        
        if self.config['super_smoothing']:
            # multi-pass smoothing for ultra-smooth lines
            for iteration in range(self.config['smoothing_iterations']):
                print(f"smoothing pass {iteration + 1}/{self.config['smoothing_iterations']}")
                
                # gaussian blur with larger kernel for smoother results
                kernel_size = 5 + (iteration * 2)  # increasing kernel size each pass
                smoothed = cv2.GaussianBlur(smoothed.astype(float), (kernel_size, kernel_size), 1.0 + iteration)
                
                # adaptive threshold to maintain line integrity
                threshold = 100 - (iteration * 10)  # decreasing threshold each pass
                smoothed = np.where(smoothed > threshold, 255, 0).astype(np.uint8)
                
                # morphological closing to maintain line connectivity
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
        else:
            # basic smoothing
            for i in range(self.config['smoothing_iterations']):
                smoothed = cv2.blur(smoothed, (3, 3))
                smoothed = np.where(smoothed > 100, 255, 0).astype(np.uint8)
        
        # final skeletonization to ensure single-pixel lines
        final_skeleton = self.advanced_thinning(smoothed)
        
        print(f"advanced smoothing complete: {np.sum(final_skeleton > 0)} pixels")
        return final_skeleton

    def advanced_thinning(self, mask):
        """advanced thinning for high-resolution images"""
        print("applying advanced thinning...")
        
        # convert to binary
        binary = (mask > 127).astype(np.uint8)
        
        # multi-pass thinning for better results at high resolution
        thinned = binary.copy()
        
        # use cross-shaped kernel for better thinning
        kernel1 = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)
        
        # diagonal kernel for second pass
        kernel2 = np.array([[1, 0, 1],
                           [0, 1, 0],
                           [1, 0, 1]], dtype=np.uint8)
        
        for i in range(self.config['thinning_iterations']):
            # alternate between kernels for better results
            kernel = kernel1 if i % 2 == 0 else kernel2
            
            # erosion followed by conditional dilation
            eroded = cv2.erode(thinned, kernel, iterations=1)
            
            # only keep pixels that maintain connectivity
            temp = cv2.dilate(eroded, kernel, iterations=1)
            thinned = cv2.bitwise_and(thinned, temp)
            thinned = cv2.bitwise_or(thinned, eroded)
        
        # ensure proper binary values
        thinned = np.where(thinned > 0, 255, 0).astype(np.uint8)
        
        return thinned

    def clean_mask(self, mask):
        """clean up the detected mask and remove isolated noise"""
        # basic morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # remove small isolated components (this is the key addition)
        cleaned = self.remove_isolated_noise(cleaned)
        
        return cleaned

    def remove_isolated_noise(self, mask):
        """remove small isolated dots and noise"""
        print("removing isolated noise dots...")
        
        # find all connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # create output mask
        clean_mask = np.zeros_like(mask)
        
        # get minimum component area from config (default 50)
        min_component_area = self.config.get('noise_removal_threshold', 50)
        components_kept = 0
        
        for i in range(1, num_labels):  # skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= min_component_area:
                # keep this component
                clean_mask[labels == i] = 255
                components_kept += 1
        
        print(f"kept {components_kept} large components (>={min_component_area} pixels), removed {num_labels - 1 - components_kept} isolated dots")
        return clean_mask

    def create_direct_lineart(self, skeleton):
        """create lineart by drawing skeleton points with anti-aliasing"""
        print("creating direct lineart from skeleton...")
        
        lineart = np.full((self.config['output_height'], self.config['output_width'], 3), 255, dtype=np.uint8)
        
        # find all skeleton points
        y_coords, x_coords = np.where(skeleton > 0)
        
        if len(y_coords) == 0:
            return lineart
        
        # calculate line thickness
        line_thickness = max(1, int(self.config['line_width']))
        
        # draw all skeleton points with anti-aliasing
        for y, x in zip(y_coords, x_coords):
            cv2.circle(lineart, (x, y), line_thickness, (0, 0, 0), -1, cv2.LINE_AA)
        
        # apply light gaussian blur for soft edges
        if self.config.get('anti_aliasing', True) and line_thickness > 1:
            lineart_gray = cv2.cvtColor(lineart, cv2.COLOR_BGR2GRAY)
            
            # light blur for soft edges
            blur_kernel = max(3, min(5, line_thickness))
            if blur_kernel % 2 == 0:
                blur_kernel += 1
                
            blurred = cv2.GaussianBlur(lineart_gray, (blur_kernel, blur_kernel), 0.6)
            lineart = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        
        print(f"created lineart with {len(y_coords)} skeleton points")
        return lineart

    def skeleton_to_svg_paths(self, skeleton):
        """convert skeleton to smooth svg paths - improved for clean curves"""
        print("converting skeleton to smooth svg paths...")
        start_time = datetime.now()
        
        # find all white pixels (skeleton points)  
        y_coords, x_coords = np.where(skeleton > 0)
        
        if len(y_coords) == 0:
            print("no skeleton points found")
            return []
        
        print(f"processing {len(y_coords)} skeleton pixels...")
        points = list(zip(x_coords, y_coords))
        
        if self.config.get('curve_smoothing', True):
            # advanced curve fitting for smooth paths
            return self.create_smooth_svg_curves(points)
        else:
            # simple linear path (original method)
            return self.create_simple_svg_path(points)

    def create_smooth_svg_curves(self, points):
        """create smooth curved SVG paths instead of jagged lines"""
        if len(points) < 4:
            return self.create_simple_svg_path(points)
        
        # sort points to create coherent paths
        points.sort(key=lambda p: (p[1] // 20, p[0]))  # group by row sections
        
        # reduce points for smoother curves (every 3rd point)
        step = max(1, len(points) // 800)  # limit to ~800 control points
        reduced_points = points[::step]
        
        if len(reduced_points) < 4:
            return self.create_simple_svg_path(reduced_points)
        
        # create smooth path using quadratic bezier curves
        path_data = f"M {reduced_points[0][0]},{reduced_points[0][1]}"
        
        # use quadratic curves for smoother results
        for i in range(1, len(reduced_points) - 1, 2):
            if i + 1 < len(reduced_points):
                # control point
                cp_x, cp_y = reduced_points[i]
                # end point  
                end_x, end_y = reduced_points[i + 1]
                path_data += f" Q {cp_x},{cp_y} {end_x},{end_y}"
            else:
                # final line if odd number of points
                x, y = reduced_points[i]
                path_data += f" L {x},{y}"
        
        print(f"created smooth curved path with {len(reduced_points)} control points")
        return [path_data]

    def create_simple_svg_path(self, points):
        """create simple linear SVG path (fallback method)"""
        if len(points) < 2:
            return []
            
        # sort points roughly
        points.sort(key=lambda p: (p[1] // 50, p[0]))
        
        # limit to reasonable number of points
        step = max(1, len(points) // 1000)
        reduced_points = points[::step]
        
        path_data = f"M {reduced_points[0][0]},{reduced_points[0][1]}"
        for x, y in reduced_points[1:]:
            path_data += f" L {x},{y}"
        
        return [path_data]

    def save_svg(self, path_strings, output_name):
        """save path strings as svg - simplified version"""
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{output_name}.svg")
        
        svg = ET.Element('svg')
        svg.set('width', str(self.config['output_width']))
        svg.set('height', str(self.config['output_height']))
        svg.set('viewBox', f"0 0 {self.config['output_width']} {self.config['output_height']}")
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        
        # white background
        bg_rect = ET.SubElement(svg, 'rect')
        bg_rect.set('width', '100%')
        bg_rect.set('height', '100%')
        bg_rect.set('fill', 'white')
        
        # add each path string
        for path_data in path_strings:
            if not path_data or len(path_data) < 10:  # skip empty or too short paths
                continue
                
            path_elem = ET.SubElement(svg, 'path')
            path_elem.set('d', path_data)
            path_elem.set('stroke', 'black')
            path_elem.set('stroke-width', str(self.config['line_width']))
            path_elem.set('fill', 'none')
            path_elem.set('stroke-linecap', 'round')
            path_elem.set('stroke-linejoin', 'round')
        
        tree = ET.ElementTree(svg)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        print(f"saved svg with {len(path_strings)} paths to {output_path}")
        return output_path

    def resize_for_display(self, img, max_width=1200, max_height=900):
        """resize for web display - maximum quality preservation"""
        h, w = img.shape[:2]
        
        # don't upscale small images, just return them as-is if smaller than max
        if w <= max_width and h <= max_height:
            print(f"image {w}×{h} kept at original size (no downscaling needed)")
            return img
        
        # calculate scale to fit within max bounds
        scale = min(max_width/w, max_height/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        print(f"resizing for display: {w}×{h} -> {new_w}×{new_h} (scale: {scale:.3f})")
        
        # use highest quality interpolation
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    def save_display_images(self, original, mask, lineart, output_name):
        """save ultra high-quality display images - lossless PNG"""
        base_name = Path(output_name).stem
        
        # use PNG for lossless quality instead of JPEG
        original_path = f'static/images/{base_name}_original.png'
        mask_path = f'static/images/{base_name}_mask.png' 
        lineart_path = f'static/images/{base_name}_lineart.png'
        
        # save as PNG for lossless quality
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # minimal compression for speed
        
        cv2.imwrite(original_path, original, png_params)
        cv2.imwrite(mask_path, mask, png_params)
        cv2.imwrite(lineart_path, lineart, png_params)
        
        print(f"saved lossless PNG images: original={original.shape}, mask={mask.shape}, lineart={lineart.shape}")
        
        return {
            'original': url_for('static', filename=f'images/{base_name}_original.png'),
            'mask': url_for('static', filename=f'images/{base_name}_mask.png'),
            'lineart': url_for('static', filename=f'images/{base_name}_lineart.png')
        }

    def save_display_images(self, original, mask, lineart, output_name):
        """save high-quality display images"""
        base_name = Path(output_name).stem
        
        original_path = f'static/images/{base_name}_original.jpg'
        mask_path = f'static/images/{base_name}_mask.jpg'
        lineart_path = f'static/images/{base_name}_lineart.jpg'
        
        # save with high quality settings
        jpg_quality = [cv2.IMWRITE_JPEG_QUALITY, 95]  # high quality JPEG
        
        cv2.imwrite(original_path, original, jpg_quality)
        cv2.imwrite(mask_path, mask, jpg_quality)
        cv2.imwrite(lineart_path, lineart, jpg_quality)
        
        print(f"saved high-quality display images: {original.shape} -> display size")
        
        return {
            'original': url_for('static', filename=f'images/{base_name}_original.jpg'),
            'mask': url_for('static', filename=f'images/{base_name}_mask.jpg'),
            'lineart': url_for('static', filename=f'images/{base_name}_lineart.jpg')
        }

# initialize generator
generator = SimpleLineartGenerator()

# error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'internal server error'}), 500

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({'error': 'file too large (max 16MB)'}), 413

# routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return jsonify({'status': 'simple lineart server working', 'timestamp': datetime.now().isoformat()})

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("upload request received")
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'no file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'no file selected'})
        
        if file:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            
            print(f"saving file to: {file_path}")
            file.save(file_path)
            
            return jsonify({
                'success': True, 
                'filename': safe_filename,
                'message': 'file uploaded successfully'
            })
            
    except Exception as e:
        print(f"upload error: {e}")
        return jsonify({'success': False, 'error': f'upload failed: {str(e)}'})

@app.route('/process', methods=['POST'])
def process_image():
    try:
        print("process request received")
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'no data provided'})
        
        filename = data.get('filename')
        config = data.get('config', {})
        
        if not filename:
            return jsonify({'success': False, 'error': 'no filename provided'})
        
        print(f"processing with simple approach: {filename}")
        generator.update_config(config)
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(input_path):
            return jsonify({'success': False, 'error': f'file not found: {filename}'})
        
        output_name = Path(filename).stem
        result = generator.process_image_simple(input_path, output_name)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"process error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'processing failed: {str(e)}'})

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename}.svg")
        if not os.path.exists(file_path):
            return jsonify({'error': f'file not found: {filename}.svg'}), 404
        
        return send_file(file_path, as_attachment=True, download_name=f"{filename}_lineart.svg")
    except Exception as e:
        print(f"download error: {e}")
        return jsonify({'error': f'download failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Professional LineArt-O-Matic Web Application")
    print("Open your browser to: http://localhost:5000")
    print("Processing pipeline: detect -> thin -> smooth -> curve generation -> SVG")
    print("Features: 4K resolution, color filtering, variable line thickness")
    print(f"Working directory: {os.getcwd()}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)