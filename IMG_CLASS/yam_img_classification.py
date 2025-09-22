from flask import Flask, Response, render_template_string, request, jsonify,url_for
from picamera2 import Picamera2
import io
import threading
import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from queue import Queue, Empty
import json
import serial
import serial.tools.list_ports
import os

app = Flask(__name__)

# Global variables
picam2 = None
frame = None
frame_lock = threading.Lock()
is_classifying = False
confidence_threshold = 0.8

# Serial communication variables (added from object detection code)
serial_queue = Queue()
serial_thread = None
serial_active = False
ser = None
serial_lock = threading.Lock()
MAX_SERIAL_RATE = 10  # Reduced rate for classification (10 messages per second)

model_path = "./models/model_float16.tflite"
labels = ['bad_yam', 'good_yam','not_yam']
interpreter = None
classification_queue = Queue(maxsize=1)

def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "USB" in port.device or "ACM" in port.device or "Arduino" in port.description:
            return port.device
    return None

arduino_port = find_arduino_port()
if arduino_port:
    ser = serial.Serial(arduino_port, baudrate=9600, timeout=1)
    print(f"Connected to Arduino at {arduino_port}")
else:
    print("Arduino not found")
    ser = None

# Serial initialization function (from object detection code)
def initialize_serial():
    global ser
    try:
        # Close existing connection if any
        if ser is not None:
            ser.close()
            time.sleep(1)  # Add delay before reopening
            
            
        # auto-detect Arduino
        arduino = next(
            (os.path.join("/dev/serial/by-id", d) for d in os.listdir("/dev/serial/by-id")
             if "usb" in d.lower() and ("ftdi" in d.lower() or "arduino" in d.lower())),
            None
        )
        if not arduino:
            print("Arduino not found")
            ser = None
            return False
            
        ser = serial.Serial(
            port=arduino,  # Use same port as object detection
            baudrate=9600,
            timeout=1,
            write_timeout=1,
            inter_byte_timeout=0.1
        )
        time.sleep(2)  # Wait for Arduino to reset
        ser.reset_input_buffer()
        return True
    except Exception as e:
        print(f"Serial init failed: {type(e).__name__}: {e}")
        ser = None
        return True

# Serial worker function (adapted from object detection code)
def serial_worker():
    global serial_active, ser
    last_heartbeat = time.time()
    heartbeat_interval = 2  # seconds
    error_count = 0
    max_error_count = 5
    last_send_time = time.time()
    MAX_SERIAL_RATE = 20  #messages per sencond

    
    while serial_active:
        try:
            # Maintain connection with error backoff
            if ser is None or not ser.is_open:
                if error_count >= max_error_count:
                    print("Max serial errors reached, cooling down...")
                    time.sleep(5)
                    error_count = 0
                
                if not initialize_serial():
                    error_count += 1
                    time.sleep(1 * error_count)  # Exponential backoff
                    continue
                    
            # Rate limiting
            min_interval = 1.0 / MAX_SERIAL_RATE
            elapsed = time.time() - last_send_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)    
            
            # Process queue with timeout
            try:
                data = serial_queue.get(timeout=0.1)
                
                # Validate data structure before sending
                if not isinstance(data, dict):
                    print("Invalid data format, skipping")
                    continue
                
                # Compact JSON with error checking
                try:
                    json_str = json.dumps(data, separators=(',', ':')) + '\n'
                    if len(json_str) > 512:  # Prevent oversized messages
                        print("Message too large, truncating")
                        json_str = json_str[:512] + '...\n'
                except (TypeError, ValueError) as e:
                    print(f"JSON encoding error: {e}")
                    continue
                
                # Safe serial write with timeout
                with serial_lock:
                    try:
                        ser.reset_input_buffer()  # Clear any pending input
                        bytes_written = ser.write(json_str.encode('utf-8'))
                        ser.flush()
                        last_heartbeat = time.time()
                        last_send_time = time.time()
                        error_count = 0  # Reset on success
                        
                        if bytes_written != len(json_str):
                            print(f"Partial write: {bytes_written}/{len(json_str)} bytes")
                            
                    except serial.SerialTimeoutException:
                        print("Write timeout, retrying...")
                        continue
                        
                    except serial.SerialException as e:
                        print(f"Serial write error: {e}")
                        ser.close()
                        ser = None
                        error_count += 1
                        continue
                        
            except Empty:
                # Send heartbeat if queue is empty
                if time.time() - last_heartbeat > heartbeat_interval:
                    with serial_lock:
                        try:
                            ser.write(b'{"heartbeat":1}\n')
                            ser.flush()
                            last_heartbeat = time.time()
                            last_send_time = time.time()
                            
                        except Exception as e:
                            print(f"Heartbeat failed: {e}")
                            ser.close()
                            ser = None
                continue
                
        except Exception as e:
            print(f"Unexpected serial error: {e}")
            error_count += 1
            if ser:
                try:
                    ser.close()
                except:
                    pass
                ser = None
            time.sleep(1)

# Start serial communication
def start_serial():
    global serial_thread, serial_active
    if serial_thread is None or not serial_thread.is_alive():
        serial_active = True
        if initialize_serial():  # Ensure connection before starting thread
            serial_thread = threading.Thread(target=serial_worker, daemon=True)
            serial_thread.start()
            print("Serial thread started")
        else:
            print("Failed to initialize serial connection")
            
def verify_serial_port():
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        print(f"Found port: {p.device} - {p.description}")
    return any('ACM0' in p.device for p in ports)


# Stop serial communication
def stop_serial():
    global serial_active
    serial_active = False

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
    print("Classification model loaded successfully")
    
    while True:
        if is_classifying:
            try:
                with frame_lock:
                    if frame is not None:
                        img = Image.open(io.BytesIO(frame))
                        predictions = classify_image(img, interpreter)
                        max_prob = np.max(predictions)
                        
                        if max_prob >= confidence_threshold:
                            label = labels[np.argmax(predictions)]
                        else:
                            label = 'Uncertain'
                        
                        # Create classification result
                        result = {
                            'label': label,
                            'probability': float(max_prob),
                            'timestamp': time.time(),
                            'all_predictions': {labels[i]: float(predictions[i]) for i in range(len(labels))}
                        }
                        
                        # Add to classification queue for web interface
                        classification_queue.put(result)
                        
                        # Send to serial queue for Arduino communication
                        # Only send if classification is confident or uncertain (not processing)
                        if label != 'Processing':
                            serial_data = {
                                'type': 'classification',
                                'result': label,
                                'confidence': float(max_prob),
                                'timestamp': time.time()
                            }
                            serial_queue.put(serial_data)
                            
            except Exception as e:
                print(f"Error in classification worker: {e}")
                import traceback
                traceback.print_exc()
                
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
                .btn-success {
                    background-color: #27ae60;
                    border-color: #27ae60;
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
                .serial-status {
                    background-color: #e8f5e8;
                    border-left: 4px solid #27ae60;
                    padding: 10px;
                    border-radius: 4px;
                    margin-top: 10px;
                    font-size: 14px;
                }
                
                .serial-status.connected {
                    background-color: #e8f5e8;
                    border-left-color: #27ae60;
                }

                .serial-status.disconnected {
                    background-color: #fdf2f2;
                    border-left-color: #e74c3c;
                }

                .status-indicator {
                    display: inline-block;
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    margin-right: 5px;
                }

                .status-connected {
                    background-color: #27ae60;
                }

                .status-disconnected {
                    background-color: #e74c3c;
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
                
                function testSerial() {
                    $.get('/test_serial', function(data) {
                        alert('Serial test: ' + JSON.stringify(data));
                    });
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
                
                function updateSerialStatus() {
                    $.get('/serial_status', function(data) {
                        let status = data.serial_active ? 'Connected' : 'Disconnected';
                        let queueSize = data.queue_size || 0;
                        $('#serialStatus').html(
                            `Serial: ${status} | Queue: ${queueSize} messages`
                        );
                    });
                }
                
                function updateSerialStatus() {
                    $.get('/serial_status', function(data) {
                        let status = (data.serial_active && data.serial_port_open) ? 'Connected' : 'Disconnected';
                        let statusColor = status === 'Connected' ? '#27ae60' : '#e74c3c';
                        let queueSize = data.queue_size || 0;
                        
                        $('#serialStatus').html(
                            `<span style="color: ${statusColor}">Arduino: ${status}</span> | Queue: ${queueSize} messages`
                        );
                        
                        // Optional: Change the background color of the status box
                        if (status === 'Connected') {
                            $('.serial-status').css('background-color', '#e8f5e8');
                            $('.serial-status').css('border-left-color', '#27ae60');
                        } else {
                            $('.serial-status').css('background-color', '#fdf2f2');
                            $('.serial-status').css('border-left-color', '#e74c3c');
                        }
                    }).fail(function() {
                        $('#serialStatus').html('<span style="color: #e74c3c">Arduino: Error</span>');
                    });
                }
              
                $(document).ready(function() {
                    setInterval(updateClassification, 100);
                    setInterval(updateSerialStatus, 1000);
                });
            </script>
        </head>
        <body>
            <h1> AI Yam  Disease Classification</h1>
            
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
                    <button onclick="testSerial()" class="btn btn-success">
                        Test Serial
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
                
                <div class="serial-status">
                    <div id="serialStatus">Serial: Checking...</div>
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

# Serial status route (added from object detection code)
@app.route('/serial_status')
def serial_status():
    return jsonify({
        'serial_thread_alive': serial_thread.is_alive() if serial_thread else False,
        'queue_size': serial_queue.qsize(),
        'serial_active': serial_active,
        'serial_port_open': ser.is_open if ser else False,
        'connection_status': 'Connected' if (ser and ser.is_open) else 'Disconnected'  # Add this line
    })

# Test serial route (added from object detection code)
@app.route('/test_serial')
def test_serial():
    test_data = {
        "type": "test",
        "message": "Classification test message",
        "timestamp": time.time()
    }
    serial_queue.put(test_data)
    return jsonify({"status": "test message queued", "data": test_data})
    
    
# Cleanup function
def cleanup():
    global picam2, is_classifying, serial_active, ser
    
    # Signal threads to stop
    is_classifying = False
    serial_active = False
    
    # Clean up camera
    if picam2:
        try:
            picam2.stop()
            picam2.close()
        except Exception as e:
            print(f"Camera cleanup error: {e}")
        finally:
            picam2 = None
    
    # Clean up serial
    if ser is not None:
        try:
            if ser.is_open:
                ser.close()
        except Exception as e:
            print(f"Serial cleanup error: {e}")
        finally:
            ser = None
    
    print("Cleanup completed")

if __name__ == '__main__':
    try:
        # Initialize serial first
        print("Initializing serial connection...")
        start_serial()
        
        # Initialize camera
        print("Initializing camera...")
        initialize_camera()
        
        # Start worker threads
        threading.Thread(target=get_frame, daemon=True).start()
        threading.Thread(target=classification_worker, daemon=True).start()
        
        # Start Flask app
        app.run(host='0.0.0.0', port=5000, threaded=True)
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        cleanup()
