// global variables for file handling
let uploadedFiles = [];
let currentFilename = '';
let processedData = null;
let colorProcessingMode = 'whole'; // 'whole' or 'ignore-color'

// initialize everything when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeSliders();
    setupFileUpload();
    setupModeButtons();
    setSimpleDefaults();
});

// set up mode buttons for color processing
function setupModeButtons() {
    const modeButtons = document.querySelectorAll('.mode-btn');
    modeButtons.forEach(btn => {
        btn.onclick = function() {
            modeButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            colorProcessingMode = this.dataset.mode;
            
            const mode = colorProcessingMode === 'whole' ? 'whole page processing' : 'ignore color mode';
            showStatus(`Switched to ${mode}`, 'success');
        };
    });
}

// set default values optimized for professional lineart
function setSimpleDefaults() {
    // set professional default values
    document.getElementById('pencilSensitivity').value = 15;
    document.getElementById('colorVariance').value = 20;
    document.getElementById('graphiteBoost').value = 12;
    document.getElementById('backgroundFilter').value = 5;
    document.getElementById('minLength').value = 8;
    document.getElementById('noiseRemoval').value = 50;  // new default
    document.getElementById('simplification').value = 0.001;
    document.getElementById('lineWidth').value = 2.0;  // medium thickness
    document.getElementById('superSmoothing').checked = true;
    document.getElementById('curveSmoothing').checked = true;
    
    // set default mode
    colorProcessingMode = 'whole';
    
    // update displays
    initializeSliders();
    showStatus('Ready for professional lineart processing', 'success');
}

// set up all the slider controls
function initializeSliders() {
    const sliders = [
        {id: 'pencilSensitivity', displayId: 'pencilValue'},
        {id: 'colorVariance', displayId: 'colorValue'},  
        {id: 'graphiteBoost', displayId: 'graphiteBoostValue'},
        {id: 'backgroundFilter', displayId: 'backgroundValue'},
        {id: 'minLength', displayId: 'lengthValue'},
        {id: 'noiseRemoval', displayId: 'noiseValue'},  // new slider
        {id: 'simplification', displayId: 'simplificationValue'},
        {id: 'lineWidth', displayId: 'lineWidthValue'}
    ];

    sliders.forEach(({id, displayId}) => {
        const slider = document.getElementById(id);
        const display = document.getElementById(displayId);
        
        if (slider && display) {
            // update display when slider changes
            slider.oninput = function() {
                display.textContent = this.value;
            };
            // set initial display value
            display.textContent = slider.value;
        } else {
            console.log(`Warning: Could not find slider ${id} or display ${displayId}`);
        }
    });
    
    // set up checkbox listeners
    const superSmoothingCheckbox = document.getElementById('superSmoothing');
    if (superSmoothingCheckbox) {
        superSmoothingCheckbox.onchange = function() {
            const status = this.checked ? 'enabled' : 'disabled';
            showStatus(`Super smoothing ${status}`, 'success');
        };
    }
    
    const curveSmoothingCheckbox = document.getElementById('curveSmoothing');
    if (curveSmoothingCheckbox) {
        curveSmoothingCheckbox.onchange = function() {
            const status = this.checked ? 'enabled' : 'disabled';
            showStatus(`Curve smoothing ${status} - creates smoother SVG paths`, 'success');
        };
    }
}

// set up file upload functionality
function setupFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');

    // handle file input change
    fileInput.onchange = function(e) {
        handleFiles(e.target.files);
    };

    // handle drag and drop events
    uploadArea.ondragover = function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    };

    uploadArea.ondragleave = function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    };

    uploadArea.ondrop = function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    };

    // handle batch mode toggle
    document.getElementById('batchMode').onchange = function() {
        const batchPanel = document.getElementById('batchPanel');
        batchPanel.classList.toggle('active', this.checked);
    };
}

// set up preset buttons
function setupPresets() {
    const presetButtons = document.querySelectorAll('.preset-btn');
    presetButtons.forEach(btn => {
        btn.onclick = function() {
            presetButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            applyPreset(this.dataset.preset);
        };
    });
}

// apply a configuration preset
function applyPreset(presetName) {
    if (presetName === 'custom') return;
    
    const preset = presets[presetName];
    if (!preset) return;

    // update slider values
    if (preset.pencil_sensitivity) document.getElementById('pencilSensitivity').value = preset.pencil_sensitivity;
    if (preset.color_variance_threshold) document.getElementById('colorVariance').value = preset.color_variance_threshold;
    if (preset.graphite_boost) document.getElementById('graphiteBoost').value = preset.graphite_boost;
    if (preset.background_filter) document.getElementById('backgroundFilter').value = preset.background_filter;
    if (preset.epsilon_factor) document.getElementById('simplification').value = preset.epsilon_factor;
    if (preset.line_width) document.getElementById('lineWidth').value = preset.line_width;

    // update display values
    initializeSliders();

    showStatus(`Applied ${presetName} preset`, 'success');
}

// handle uploaded files
function handleFiles(files) {
    uploadedFiles = Array.from(files);
    const isBatch = document.getElementById('batchMode').checked;

    if (isBatch) {
        updateBatchFileList();
        document.getElementById('batchProcessBtn').disabled = false;
    } else {
        if (files.length > 0) {
            uploadSingleFile(files[0]);
        }
    }
}

// upload a single file to the server
function uploadSingleFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    showStatus('Uploading file...', 'processing');

    // show image preview immediately
    showImagePreview(file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        // check if response is ok before trying to parse json
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        return response.text(); // get as text first
    })
    .then(text => {
        // try to parse json, handle empty responses
        if (!text.trim()) {
            throw new Error('Empty response from server');
        }
        const data = JSON.parse(text);
        
        if (data.success) {
            currentFilename = data.filename;
            document.getElementById('processBtn').disabled = false;
            showStatus(`File uploaded: ${file.name} - Ready to process`, 'success');
        } else {
            showStatus(`Upload failed: ${data.error}`, 'error');
        }
    })
    .catch(error => {
        console.error('Upload error:', error);
        showStatus(`Upload error: ${error.message}`, 'error');
    });
}

// show image preview before processing
function showImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewDiv = document.getElementById('imagePreview');
        const previewImg = document.getElementById('previewImg');
        const imageInfo = document.getElementById('imageInfo');
        
        previewImg.src = e.target.result;
        imageInfo.textContent = `File: ${file.name} | Size: ${(file.size / (1024*1024)).toFixed(2)} MB | Type: ${file.type}`;
        previewDiv.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// update the batch file list display
function updateBatchFileList() {
    const listDiv = document.getElementById('batchFileList');
    listDiv.innerHTML = '<h4>📋 Files to Process:</h4>';
    
    uploadedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.innerHTML = `<p>${index + 1}. ${file.name}</p>`;
        listDiv.appendChild(fileItem);
    });
}

// process the uploaded image
function processImage() {
    if (!currentFilename) {
        showStatus('Please upload a file first', 'error');
        return;
    }

    const config = getConfiguration();
    const startTime = Date.now();

    // start timer display
    showProcessingProgress(startTime);
    document.getElementById('processBtn').disabled = true;

    fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            filename: currentFilename,
            config: config
        })
    })
    .then(response => {
        // check if response is ok before trying to parse json
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        return response.text(); // get as text first
    })
    .then(text => {
        // try to parse json, handle empty responses
        if (!text.trim()) {
            throw new Error('Empty response from server');
        }
        const data = JSON.parse(text);
        const processTime = ((Date.now() - startTime) / 1000).toFixed(1);
        
        // stop progress timer
        clearInterval(window.progressTimer);
        
        if (data.success) {
            processedData = data;
            displayResults(data);
            updateStats(data.stats, processTime);
            document.getElementById('downloadBtn').disabled = false;
            showStatus(`Processing complete in ${processTime}s! 🎉`, 'success');
        } else {
            showStatus(`Processing failed: ${data.error}`, 'error');
        }
        
        document.getElementById('processBtn').disabled = false;
    })
    .catch(error => {
        // stop progress timer on error
        clearInterval(window.progressTimer);
        console.error('Processing error:', error);
        showStatus(`Processing error: ${error.message}`, 'error');
        document.getElementById('processBtn').disabled = false;
    });
}

// show real-time processing progress with timer
function showProcessingProgress(startTime) {
    showStatus('Processing image... ⏱️ 0.0s', 'processing');
    
    // update timer every 100ms
    window.progressTimer = setInterval(() => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        showStatus(`Processing image... ⏱️ ${elapsed}s`, 'processing');
    }, 100);
}

// batch processing function
function processBatch() {
    // basic implementation for batch processing
    showStatus('Batch processing not yet implemented in demo', 'error');
}

// get current configuration from UI
function getConfiguration() {
    const resolution = document.getElementById('resolution').value.split('x');
    
    return {
        pencil_sensitivity: parseInt(document.getElementById('pencilSensitivity').value),
        color_variance_threshold: parseInt(document.getElementById('colorVariance').value),
        graphite_boost: parseInt(document.getElementById('graphiteBoost').value),
        background_filter: parseInt(document.getElementById('backgroundFilter').value),
        noise_removal_threshold: parseInt(document.getElementById('noiseRemoval').value),  // new setting
        line_width: parseFloat(document.getElementById('lineWidth').value),
        output_width: parseInt(resolution[0]),
        output_height: parseInt(resolution[1]),
        smoothing_iterations: Math.max(2, parseInt(document.getElementById('simplification').value * 4000)),
        thinning_iterations: 2,
        super_smoothing: document.getElementById('superSmoothing').checked,
        curve_smoothing: document.getElementById('curveSmoothing').checked,
        color_processing_mode: colorProcessingMode, // add color mode
        anti_aliasing: true
    };
}

// display processing results
function displayResults(data) {
    const resultsGrid = document.getElementById('resultsGrid');
    
    document.getElementById('originalImg').src = data.display_images.original;
    document.getElementById('maskImg').src = data.display_images.mask;
    document.getElementById('lineartImg').src = data.display_images.lineart;
    
    resultsGrid.style.display = 'grid';
}

// update statistics display
function updateStats(stats, processTime) {
    document.getElementById('contoursFound').textContent = stats.contours_found;
    document.getElementById('svgPaths').textContent = stats.svg_paths;
    document.getElementById('resolution').textContent = stats.resolution;
    document.getElementById('processTime').textContent = processTime + 's';
    
    document.getElementById('statsPanel').style.display = 'block';
}

// download the generated svg file
function downloadSVG() {
    if (!currentFilename || !processedData) {
        showStatus('No processed file to download', 'error');
        return;
    }

    const filename = currentFilename.split('.')[0];
    window.open(`/download/${filename}`, '_blank');
}

// reset all form data and results
function resetAll() {
    uploadedFiles = [];
    currentFilename = '';
    processedData = null;
    
    // stop any running timers
    if (window.progressTimer) {
        clearInterval(window.progressTimer);
    }
    
    document.getElementById('fileInput').value = '';
    document.getElementById('processBtn').disabled = true;
    document.getElementById('downloadBtn').disabled = true;
    document.getElementById('batchProcessBtn').disabled = true;
    
    document.getElementById('resultsGrid').style.display = 'none';
    document.getElementById('statsPanel').style.display = 'none';
    document.getElementById('statusArea').innerHTML = '';
    document.getElementById('batchFileList').innerHTML = '';
    
    // hide image preview
    document.getElementById('imagePreview').style.display = 'none';
    
    // reset mode buttons
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector('.mode-btn[data-mode="whole"]').classList.add('active');
    colorProcessingMode = 'whole';
    
    setSimpleDefaults();
    showStatus('Reset to professional defaults', 'success');
}

// show status messages to user
function showStatus(message, type) {
    const statusArea = document.getElementById('statusArea');
    const status = document.createElement('div');
    status.className = `status ${type}`;
    status.textContent = message;
    
    statusArea.innerHTML = '';
    statusArea.appendChild(status);
    
    // auto-remove success messages after 3 seconds
    if (type === 'success') {
        setTimeout(() => {
            if (statusArea.contains(status)) {
                statusArea.removeChild(status);
            }
        }, 3000);
    }
}

// show full size image in modal
function showFullImage(imageSrc) {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImg');
    
    modal.style.display = 'block';
    modalImg.src = imageSrc;
    
    // prevent body scrolling when modal is open
    document.body.style.overflow = 'hidden';
}

// hide full size image modal
function hideFullImage() {
    const modal = document.getElementById('imageModal');
    modal.style.display = 'none';
    
    // restore body scrolling
    document.body.style.overflow = 'auto';
}

// close modal with escape key
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        hideFullImage();
    }
});