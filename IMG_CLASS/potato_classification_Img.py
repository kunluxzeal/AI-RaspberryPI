#!/home/Jimmy/tflite/bin/python

from flask import Flask, Response, render_template_string, request, jsonify,url_for
from picamera2 import Picamera2
import io
import threading
import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from queue import Queue , Empty

app = Flask(__name__)

# Global variables
picam2 = None
frame = None
frame_lock = threading.Lock()
is_classifying = False
confidence_threshold = 0.8

model_path = "./models/img_class_model/model_float16.tflite"
#model_path = "./models/img_class_model/ei-img_class-transfer-learning-tensorflow-lite-float32-model.3.ilte"
labels = ['Not_Potato', 'Potato_Early_blight', 'Potato_Late_blight','Potato_Healthy','Unknown']
interpreter = None
classification_queue = Queue(maxsize=1)

def initialize_camera():
	
	global picam2
	picam2 = Picamera2()
	config = picam2.create_preview_configuration(main={"size": (320, 240)})
	picam2.configure(config)
	picam2.start()
	time.sleep(2) # Wait for camera to warm up

def get_frame():
	
	global frame
	while True:
		stream = io.BytesIO()
		picam2.capture_file(stream, format='jpeg')
		with frame_lock:
			frame = stream.getvalue()
		time.sleep(0.1) # Capture frames more frequently

def generate_frames():
	
	while True:
		with frame_lock:
			if frame is not None:
				yield (b'--frame\r\n'
						b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
		time.sleep(0.1)

def load_model():
	
	global interpreter
	if interpreter is None:
		interpreter = tflite.Interpreter(model_path=model_path)
		interpreter.allocate_tensors()
	return interpreter


def classify_image(img, interpreter):
		
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	
	img = img.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
	input_data = np.expand_dims(np.array(img), axis=0).astype(input_details[0]['dtype'])
	
	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()
	
	predictions = interpreter.get_tensor(output_details[0]['index'])[0]
	
	# Handle output based on type
	output_dtype = output_details[0]['dtype']
	
	if output_dtype in [np.int8, np.uint8]:
		# Dequantize the output
		scale, zero_point = output_details[0]['quantization']
		predictions = (predictions.astype(np.float32) - zero_point) * scale
	return predictions
		
def classification_worker():
	interpreter = load_model()
	while True:
		if is_classifying:
			with frame_lock:
				if frame is not None:
					img = Image.open(io.BytesIO(frame))
				predictions = classify_image(img, interpreter)
				max_prob = np.max(predictions)
				if max_prob >= confidence_threshold:
					label = labels[np.argmax(predictions)]
				else:
					label = 'Uncertain'
				classification_queue.put({'label': label,'probability': float(max_prob)})
			time.sleep(1) # Adjust based on your needs
@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Image Classification</title>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {
                    font-family: 'Roboto', sans-serif;
                    background-color: #f5f7fa;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 30px;
                    font-weight: 500;
                }
                .video-container {
                    background-color: #000;
                    border-radius: 8px;
                    overflow: hidden;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }
                .controls {
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    margin-bottom: 20px;
                }
                .btn-group {
                    display: flex;
                    gap: 10px;
                    margin-bottom: 20px;
                }
                .btn {
                    flex: 1;
                    padding: 10px;
                    font-weight: 500;
                    border-radius: 6px;
                    transition: all 0.2s;
                }
                .btn-primary {
                    background-color: #3498db;
                    border-color: #3498db;
                }
                .btn-danger {
                    background-color: #e74c3c;
                    border-color: #e74c3c;
                }
                .btn:disabled {
                    opacity: 0.6;
                }
                .confidence-control {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-bottom: 20px;
                }
                .confidence-control input {
                    width: 80px;
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                .result-box {
                    background-color: #f8f9fa;
                    border-left: 4px solid #3498db;
                    padding: 15px;
                    border-radius: 4px;
                    font-size: 18px;
                    min-height: 60px;
                }
                .result-label {
                    font-weight: 500;
                    color: #2c3e50;
                }
                .result-value {
                    font-weight: 400;
                    color: #7f8c8d;
                }
            </style>
            <script>
                function startClassification() {
                    $.post('/start');
                    $('#startBtn').prop('disabled', true);
                    $('#stopBtn').prop('disabled', false);    
                }
                
                function stopClassification() {
                    $.post('/stop');
                    $('#startBtn').prop('disabled', false);
                    $('#stopBtn').prop('disabled', true);
                }
                
                function updateConfidence() {
                    var confidence = $('#confidence').val();
                    $.post('/update_confidence', {confidence: confidence});
                }
                
                function updateClassification() {
                    $.get('/get_classification', function(data) {
                        if (data.label && data.probability) {
                            $('#classification').html(
                                `<span class="result-label">${data.label}:</span> ` +
                                `<span class="result-value">${data.probability.toFixed(2)}</span>`
                            );
                        }
                    });
                }
                
                $(document).ready(function() {
                    setInterval(updateClassification, 100);
                });
            </script>
        </head>
        <body>
            <h1>Potato Disease Classification</h1>
            
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" width="640" height="480" />
            </div>
            
            <div class="controls">
                <div class="btn-group">
                    <button id="startBtn" onclick="startClassification()" class="btn btn-primary">
                        Start Classification
                    </button>
                    <button id="stopBtn" onclick="stopClassification()" disabled class="btn btn-danger">
                        Stop Classification
                    </button>
                </div>
                
                <div class="confidence-control">
                    <label for="confidence">Confidence Threshold:</label>
                    <input type="number" id="confidence" name="confidence" 
                           min="0" max="1" step="0.1" value="0.8" onchange="updateConfidence()">
                </div>
                
                <div class="result-box">
                    <div id="classification">Waiting for classification...</div>
                </div>
            </div>
        </body>
        </html>
    ''')
		
		
@app.route('/video_feed')
def video_feed():
	return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_classification():
	global is_classifying
	is_classifying = True
	return '', 204
	
@app.route('/stop', methods=['POST'])
def stop_classification():
	global is_classifying
	is_classifying = False
	return '', 204
	
@app.route('/update_confidence', methods=['POST'])
def update_confidence():
	global confidence_threshold
	confidence_threshold = float(request.form['confidence'])
	return '', 204
	
@app.route('/get_classification')
def get_classification():
	if not is_classifying:
		return jsonify({'label': 'Not classifying', 'probability': 0})
	try:
		result = classification_queue.get_nowait()
	except Empty:
		result = {'label': 'Processing', 'probability': 0}
	return jsonify(result)
	
if __name__ == '__main__':
	initialize_camera()
	threading.Thread(target=get_frame, daemon=True).start()
	threading.Thread(target=classification_worker, daemon=True).start()
	app.run(host='0.0.0.0', port=5000, threaded=True)
