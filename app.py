# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import warnings

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Initialize model
model_dir = "./resources/anti_spoof_models"
device_id = 0
model_test = AntiSpoofPredict(device_id)
image_cropper = CropImage()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Get face bounding boxes (all detected faces)
        try:
            image_bboxes = model_test.get_bbox(image)
        except Exception as e:
            print(f"Error getting bbox: {e}")
            return jsonify({'error': f'Face detection failed: {str(e)}'}), 400
        
        # If no faces detected, return empty results
        if len(image_bboxes) == 0:
            return jsonify({'faces': []})
        
        # Process each face
        faces_results = []
        for image_bbox in image_bboxes:
            # Run prediction on all models for this face
            prediction = np.zeros((1, 3))
            for model_name in os.listdir(model_dir):
                try:
                    h_input, w_input, model_type, scale = parse_model_name(model_name)
                    param = {
                        "org_img": image,
                        "bbox": image_bbox,
                        "scale": scale,
                        "out_w": w_input,
                        "out_h": h_input,
                        "crop": True,
                    }
                    if scale is None:
                        param["crop"] = False
                    img = image_cropper.crop(**param)
                    prediction += model_test.predict(img, os.path.join(model_dir, model_name))
                except Exception as e:
                    print(f"Error processing model {model_name}: {e}")
                    continue
            
            # Get final prediction for this face
            label = np.argmax(prediction)
            value = prediction[0][label] / 2
            
            face_result = {
                'label': int(label),
                'is_real': bool(label == 1),
                'confidence': float(value),
                'message': 'Real Face' if label == 1 else 'Fake Face',
                'bbox': image_bbox
            }
            faces_results.append(face_result)
        
        result = {
            'faces': faces_results
        }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        print(f"Error in predict: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Silent Face Anti-Spoofing Demo</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 600px;
            width: 100%;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 20px;
            font-size: 24px;
        }
        .video-container {
            position: relative;
            width: 100%;
            aspect-ratio: 3/4;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        video {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        #overlayCanvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        .result-box {
            position: absolute;
            top: 10px;
            left: 10px;
            right: 10px;
            padding: 15px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            color: white;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        }
        .result-box.real {
            background: rgba(76, 175, 80, 0.8);
        }
        .result-box.fake {
            background: rgba(244, 67, 54, 0.8);
        }
        .fps-counter {
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 14px;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        button {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        #startBtn {
            background: #4CAF50;
            color: white;
        }
        #startBtn:hover {
            background: #45a049;
        }
        #stopBtn {
            background: #f44336;
            color: white;
        }
        #stopBtn:hover {
            background: #da190b;
        }
        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 2px solid #f5c6cb;
        }
        .info-message {
            background: #d1ecf1;
            color: #0c5460;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 2px solid #bee5eb;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Silent Face Anti-Spoofing</h1>
        <div id="errorMessage" class="error-message" style="display: none;"></div>
        <div id="infoMessage" class="info-message" style="display: none;"></div>
        <div class="video-container">
            <video id="video" autoplay playsinline></video>
            <canvas id="overlayCanvas"></canvas>
            <div class="overlay">
                <div id="resultBox" class="result-box" style="display: none;"></div>
                <div id="fpsCounter" class="fps-counter" style="display: none;">FPS: 0</div>
            </div>
        </div>
        <div class="controls">
            <button id="startBtn">Start Real-time Detection</button>
            <button id="stopBtn">Stop</button>
        </div>
    </div>

    <script>
        const video = document.getElementById('video');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const resultBox = document.getElementById('resultBox');
        const fpsCounter = document.getElementById('fpsCounter');
        const errorMessage = document.getElementById('errorMessage');
        const infoMessage = document.getElementById('infoMessage');
        const overlayCanvas = document.getElementById('overlayCanvas');
        const overlayCtx = overlayCanvas.getContext('2d');

        let stream = null;
        let isProcessing = false;
        let animationId = null;
        let frameCount = 0;
        let lastFpsUpdate = Date.now();
        let currentFps = 0;

        // Check if getUserMedia is supported
        function checkCameraSupport() {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                errorMessage.innerHTML = `
                    <strong>Camera not supported</strong><br>
                    Your browser doesn't support camera access or the page is not served over HTTPS.<br><br>
                    <strong>Solutions:</strong><br>
                    1. Access via localhost (http://localhost:5000)<br>
                    2. Use HTTPS (required for camera access on non-localhost)<br>
                    3. Try a different browser (Chrome/Firefox recommended)
                `;
                errorMessage.style.display = 'block';
                startBtn.disabled = true;
                return false;
            }
            return true;
        }

        startBtn.addEventListener('click', async () => {
            if (!checkCameraSupport()) return;

            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        facingMode: 'user',
                        width: { ideal: 640 },
                        height: { ideal: 480 }
                    } 
                });
                video.srcObject = stream;
                
                // Wait for video to be ready
                video.onloadedmetadata = () => {
                    // Set canvas size to match video
                    overlayCanvas.width = video.videoWidth;
                    overlayCanvas.height = video.videoHeight;
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    errorMessage.style.display = 'none';
                    
                    // Start real-time processing
                    isProcessing = true;
                    processFrame();
                };
                
            } catch (err) {
                errorMessage.innerHTML = `
                    <strong>Error accessing camera:</strong> ${err.message}<br><br>
                    <strong>Possible solutions:</strong><br>
                    1. Allow camera permission when prompted<br>
                    2. Make sure no other app is using the camera<br>
                    3. Try refreshing the page<br>
                    4. Access via localhost: http://localhost:5000
                `;
                errorMessage.style.display = 'block';
            }
        });

        stopBtn.addEventListener('click', () => {
            stopProcessing();
        });

        function stopProcessing() {
            isProcessing = false;
            if (animationId) {
                cancelAnimationFrame(animationId);
                animationId = null;
            }
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
                stream = null;
            }
            startBtn.disabled = false;
            stopBtn.disabled = true;
            resultBox.style.display = 'none';
            fpsCounter.style.display = 'none';
            // Clear canvas
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        }

        function drawBoundingBox(bbox, isReal, confidence) {
            if (!bbox || bbox.length !== 4) return;
            
            const [x, y, width, height] = bbox;
            
            // Set color based on result
            const color = isReal ? '#00FF00' : '#FF0000';
            const lineWidth = 3;
            
            // Draw rectangle
            overlayCtx.strokeStyle = color;
            overlayCtx.lineWidth = lineWidth;
            overlayCtx.strokeRect(x, y, width, height);
            
            // Draw label background
            overlayCtx.fillStyle = color;
            overlayCtx.fillRect(x, y - 25, width, 25);
            
            // Draw label text
            overlayCtx.fillStyle = '#FFFFFF';
            overlayCtx.font = 'bold 16px Arial';
            const label = isReal ? 'REAL' : 'FAKE';
            const confText = `${(confidence * 100).toFixed(0)}%`;
            overlayCtx.fillText(`${label} ${confText}`, x + 5, y - 7);
        }

        function drawAllBoundingBoxes(faces) {
            // Clear previous drawings
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            
            // Draw each face
            faces.forEach(face => {
                drawBoundingBox(face.bbox, face.is_real, face.confidence);
            });
        }

        async function processFrame() {
            if (!isProcessing || !stream) return;

            // Create canvas for frame capture
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            
            // Get image data
            const imageData = canvas.toDataURL('image/jpeg', 0.7);
            
            // Send to backend
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ image: imageData })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    console.error('Prediction error:', data.error);
                } else if (data.faces && data.faces.length > 0) {
                    // Update result display for multiple faces
                    const realCount = data.faces.filter(f => f.is_real).length;
                    const fakeCount = data.faces.filter(f => !f.is_real).length;
                    
                    resultBox.className = 'result-box ' + (realCount > 0 ? 'real' : 'fake');
                    resultBox.innerHTML = `
                        ${data.faces.length} Face(s) Detected<br>
                        <small>Real: ${realCount} | Fake: ${fakeCount}</small>
                    `;
                    resultBox.style.display = 'block';
                    
                    // Draw all bounding boxes
                    drawAllBoundingBoxes(data.faces);
                    
                    // Update FPS counter
                    frameCount++;
                    const now = Date.now();
                    if (now - lastFpsUpdate >= 1000) {
                        currentFps = frameCount;
                        frameCount = 0;
                        lastFpsUpdate = now;
                        fpsCounter.textContent = `FPS: ${currentFps}`;
                        fpsCounter.style.display = 'block';
                    }
                } else {
                    // No faces detected
                    resultBox.className = 'result-box';
                    resultBox.innerHTML = 'No faces detected';
                    resultBox.style.display = 'block';
                    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
                }
            } catch (err) {
                console.error('Request error:', err);
            }

            // Continue processing loop
            if (isProcessing) {
                animationId = requestAnimationFrame(processFrame);
            }
        }

        // Initialize button states
        stopBtn.disabled = true;
        
        // Check camera support on load
        window.addEventListener('load', () => {
            if (!checkCameraSupport()) {
                infoMessage.innerHTML = `
                    <strong>For best results, access this page via:</strong><br>
                    • http://localhost:5000 (if on the same machine)<br>
                    • http://192.168.1.12:5000 (if on the same network)<br><br>
                    <strong>Note:</strong> Camera access requires HTTPS for non-localhost addresses.
                `;
                infoMessage.style.display = 'block';
            }
        });
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
