from flask import Flask, Response, render_template_string, request, jsonify
from picamera2 import Picamera2
import io
import threading
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite
from queue import Queue, Empty
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import signal
import json
import serial

MAX_SERIAL_RATE = 20  # messages per second
last_send_time = 0


#Add to global variables
serial_queue = Queue()
serial_thread = None
serial_active = False



app = Flask(__name__)

# Global variables
picam2 = None
frame = None
frame_lock = threading.Lock()
is_detecting = False
confidence_threshold = 0.5

ser = None
serial_lock = threading.Lock()

model_path = "./obj_model/ei-object-detection-object-detection-tensorflow-lite-int8-quantized-model.3.lite"
labels_path = "./obj_model/label.txt"
#labels_path = "./models/coco_labels.txt"

interpreter = None
detection_queue = Queue(maxsize=10)
latest_detections = []
detections_lock = threading.Lock()

def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = load_labels(labels_path)

def initialize_camera():
    global picam2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Wait for camera to warm up
    


#Serial initialization with Arduino
def initialize_serial():
    global ser
    try:
        # Close existing connection if any
        if ser is not None:
            ser.close()
            time.sleep(1)  # Add delay before reopening
            
        ser = serial.Serial(
            port='/dev/ttyACM0',
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
        return False

def get_frame():
    global frame
    frame_interval = 0.3  # Reduce frame rate from 10fps to ~3fps
    while True:
        start_time = time.time()
        
        # Capture frame
        stream = io.BytesIO()
        picam2.capture_file(stream, format='jpeg')
        stream.seek(0)
        img = Image.open(stream)
        
        # Only process if detecting
        if is_detecting:
            img = draw_detections(img)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        
        with frame_lock:
            frame = img_byte_arr.getvalue()
        
        # Maintain frame interval
        elapsed = time.time() - start_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

def generate_frames():
    while True:
        with frame_lock:
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # Adjust this value to control frame rate

def load_model():
    global interpreter
    if interpreter is None:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    return interpreter

def detect_objects(img, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get the expected input shape
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]

    # Resize and preprocess the image
    img = img.resize((width, height))
    img = img.convert('RGB')  # Ensure the image is in RGB format
    input_data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)
    
    

    # Check if the model expects float input
    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    elif input_details[0]['dtype'] == np.int8:
        scale , zero_point = input_details[0]['quantization']
        input_data = (np.array(input_data, dtype = np.float32) / scale +zero_point).astype(np.int8)

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()
    
    for i, detail in enumerate(output_details):
        tensor_data = interpreter.get_tensor(detail['index'])
        print(f"Output {i} tensor data shape: {tensor_data.shape}")
        print(f"Output {i} tensor data sample: {tensor_data}")
        

    # Get outputs
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[2]['index'])[0])

    return boxes, classes, scores, num_detections
    


def detection_worker():
    global frame, is_detecting, latest_detections
    interpreter = load_model()
    print("Model loaded successfully")
    pixel_size_um = 400  # micrometers
    pixel_size_cm = pixel_size_um / 10_000  # cm


    while True:
        if is_detecting:
            try:
                current_frame = None
                with frame_lock:
                    if frame is not None:
                        current_frame = frame

                if current_frame is not None:
                    img = Image.open(io.BytesIO(current_frame))
                    
                    boxes, classes, scores, num_detections = detect_objects(img, interpreter)
                    
                    new_detections = []
                    for i in range(int(num_detections)):
                        if scores[i] >= confidence_threshold:
                            ymin, xmin, ymax, xmax = boxes[i]
                            (left, right, top, bottom) = (
                            xmin * img.width, xmax * img.width,
							ymin * img.height, ymax * img.height
							)
                            box_width_px = right - left
                            box_height_px = bottom - top
                            
                            box_width_cm_sensor = box_width_px * pixel_size_cm
                            box_height_cm_sensor = box_height_px * pixel_size_cm
							
							
                            class_id = int(classes[i])
                            class_name = labels.get(class_id, f"Class {class_id}")
                            
							
                            
                            detection_data = {
                                'class': class_name,
                                'score': float(scores[i]),
                                'box': [left, top, right, bottom],
								'box_width': box_width_px,
                                'box_height': box_height_px,
                                'box_width_cm_sensor': box_width_cm_sensor,   # using sensor pixel size
                                'box_height_cm_sensor': box_height_cm_sensor,
                                'timestamp':time.time()
                            }
                            
                            new_detections.append(detection_data)
                            
                           
                    
                    with detections_lock:
                        latest_detections = new_detections  # Replace instead of append
                        # Send to serial queue if there are detections
                         # Send to serial queue if there are detections
                        if new_detections:
                            serial_queue.put({
                                'timestamp': time.time(),
                                'detections': new_detections
                            })

            except Exception as e:
                print(f"Error in detection worker: {e}")
                import traceback
                traceback.print_exc()
        time.sleep(0.1)  # Process frames more frequently
        
        
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
                    
                # Ensure proper JSON structure
                if 'detections' not in data:
                    if 'class' in data:
                        data = {'detections': [data]}
                    else:
                        print("Invalid detection format, skipping")
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
                        last_send_time = time.time()  # update send time 
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

def stop_serial():
    global serial_active
    serial_active = False
            

def draw_detections(img):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    with detections_lock:
        for detection in latest_detections:
            left, top, right, bottom = detection['box']
            draw.rectangle([left, top, right, bottom], outline="blue", width=2)
            
            label = f"{detection['class']}: {detection['score']:.2f}"
            draw.text((left, top-25), label, font=font, fill="blue")

    return img
    

    
@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Object Detection</title>
            <!-- Include jQuery -->
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <!-- Custom CSS styles -->
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f0f2f5;
                    margin: 20px;
                    color: #333;
                }

                h1 {
                    text-align: center;
                    color: #2c3e50;
                }

                /* Container for the main content */
                .container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    max-width: 700px;
                    margin: auto;
                    background-color: #ffffff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }

                /* Video styling */
                .video-container {
                    position: relative;
                    width: 640px;
                    height: 480px;
                    border: 2px solid #ccc;
                    border-radius: 4px;
                    overflow: hidden;
                    margin-bottom: 15px;
                }

                /* Buttons styling */
                .buttons {
                    display: flex;
                    gap: 10px;
                    margin-bottom: 15px;
                }

                button {
                    padding: 10px 20px;
                    font-size: 14px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }

                button:disabled {
                    background-color: #ccc;
                    cursor: not-allowed;
                }

                /* Button colors */
                #startBtn {
                    background-color: #27ae60;
                    color: white;
                }

                #startBtn:hover:not(:disabled) {
                    background-color: #2ecc71;
                }

                #stopBtn {
                    background-color: #c0392b;
                    color: white;
                }

                #stopBtn:hover:not(:disabled) {
                    background-color: #e74c3c;
                }

                /* Close button styling */
                button:nth-of-type(3) {
                    background-color: #2980b9;
                    color: white;
                }

                button:nth-of-type(3):hover {
                    background-color: #3498db;
                }

                /* Confidence threshold input styling */
                .confidence-container {
                    margin-bottom: 15px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                
                

                input[type=number] {
                    width: 80px;
                    padding: 5px 8px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    font-size: 14px;
                }

                /* Detections display area */
                #detections {
                    width: 100%;
                    max-height: 200px;
                    overflow-y: auto;
                    background-color: #ecf0f1;
                    padding: 10px;
                    border-radius: 4px;
                    font-size: 14px;
                }

                /* Responsive adjustments */
                @media(max-width: 700px) {
                    .container {
                        width: 90%;
                        padding: 10px;
                    }
                    .video-container {
                        width: 100%;
                        height: auto;
                    }
                }
            </style>
            <!-- Your existing scripts -->
            <script>
                function startDetection() {
                    $.post('/start');
                    $('#startBtn').prop('disabled', true);
                    $('#stopBtn').prop('disabled', false);
                }

                function stopDetection() {
                    $.post('/stop');
                    $('#startBtn').prop('disabled', false);
                    $('#stopBtn').prop('disabled', true);
                }

                function updateConfidence() {
                    var confidence = $('#confidence').val();
                    $.post('/update_confidence', {confidence: confidence});
                }
                
				function getImageMetadata() {
					$.get('/get_image_metadata', function(data) {
						console.log("Image Metadata:", data);
					alert(JSON.stringify(data, null, 2));
				});
			}


                function updateDetections() {
                    $.get('/get_detections', function(data) {
                        $('#detections').empty();
                        const sortedDetections = data.sort((a, b) => b.score - a.score);
                        sortedDetections.forEach(detection => {
                            $('#detections').append(`<p>${detection.class}: ${detection.score.toFixed(2)} |
                             Width: ${detection.box_width_cm_sensor.toFixed(3)} cm | 
							 Height: ${detection.box_height_cm_sensor.toFixed(3)} cm</p>`);
                        });
                    });
                }

                function closeApp() {
                    if (confirm('Are you sure you want to close the app?')) {
                        $.ajax({
                            url: '/close',
                            type: 'POST',
                            headers: {
                                'X-Requested-With': 'XMLHttpRequest'
                            },
                            success: function(data) {
                                alert(data);
                                window.close();
                            },
                            error: function() {
                                alert('Server has shut down. You can close this window.');
                                window.close();
                            }
                        });
                    }
                }

                $(document).ready(function() {
                    setInterval(updateDetections, 500);  // Update every 500ms
                });
            </script>
        </head>
        <body>
            <div class="container">
                <h1>Object Detection</h1>
                <div class="video-container">
                    <img src="{{ url_for('video_feed') }}" width="640" height="480" />
                </div>
                <div class="buttons">
                    <button id="startBtn" onclick="startDetection()">Start Detection</button>
                    <button id="stopBtn" onclick="stopDetection()" disabled>Stop Detection</button>
                    <button onclick="closeApp()">Close App</button>
                     <button onclick="getImageMetadata()">Get Image Metadata</button>
                </div>
                <div class="confidence-container">
                    <label for="confidence">Confidence Threshold:</label>
                    <input type="number" id="confidence" name="confidence" min="0" max="1" step="0.1" value="0.5" onchange="updateConfidence()">
                </div>
                <div id="detections">Waiting for detections...</div>
            </div>
        </body>
        </html>
    ''')
    
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_detection():
    global is_detecting
    is_detecting = True
    return '', 204

@app.route('/stop', methods=['POST'])
def stop_detection():
    global is_detecting
    is_detecting = False
    return '', 204

@app.route('/update_confidence', methods=['POST'])
def update_confidence():
    global confidence_threshold
    confidence_threshold = float(request.form['confidence'])
    return '', 204

@app.route('/get_detections')
def get_detections():
    global latest_detections
    if not is_detecting:
        return jsonify([])
    with detections_lock:
        return jsonify(latest_detections)

@app.route('/close', methods=['POST'])
def close_app():
    global is_detecting
    is_detecting = False
    cleanup()
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # This is an AJAX request, so it's an intentional close
        def shutdown():
            os.kill(os.getpid(), signal.SIGINT)
        threading.Thread(target=shutdown).start()
        return 'Server shutting down...', 200
    else:
        # This is not an AJAX request, so it's probably a page refresh
        return 'Page refreshed', 200
        
@app.route('/get_image_metadata')
def get_image_metadata():
    with frame_lock:
        if frame is not None:   
            img = Image.open(io.BytesIO(frame))
            metadata = {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format
            }
            return jsonify(metadata)
    return jsonify({"error": "No frame available"}), 404


@app.route('/serial_status')
def serial_status():
    return {
        'serial_thread_alive': serial_thread.is_alive() if serial_thread else False,
        'queue_size': serial_queue.qsize(),
        'serial_active': serial_active,
        'serial_port_open': ser.is_open if ser else False
    }

@app.route('/test_serial')
def test_serial():
    test_data = {"test": "message", "value": 123}
    serial_queue.put(test_data)
    return jsonify({"status": "test message queued", "data": test_data})
    
    
@app.route('/serial_debug')
def serial_debug():
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    return jsonify({
        'available_ports': [p.device for p in ports],
        'current_port': '/dev/ttyACM0',
        'connection_status': {
            'is_open': ser.is_open if ser else False,
            'port': ser.port if ser else None
        },
        'queue_status': {
            'size': serial_queue.qsize(),
            'empty': serial_queue.empty()
        }
    })
    
    
@app.route('/system_status')
def system_status():
    import psutil
    return jsonify({
        'cpu_usage': psutil.cpu_percent(),
        'memory': psutil.virtual_memory().percent,
        'threads': {
            'frame': threading.active_count(),
            'serial_worker': serial_thread.is_alive() if serial_thread else False,
            'detection_worker': detection_thread.is_alive() if detection_thread else False
        },
        'serial_queue': serial_queue.qsize(),
        'camera_status': picam2 is not None
    })

@app.route('/test_serial_connection')
def test_serial_connection():
    if ser and ser.is_open:
        try:
            test_msg = {"test": "connection", "timestamp": time.time()}
            serial_queue.put(test_msg)
            return jsonify({"status": "test message queued"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Serial not connected"}), 400

def cleanup():
    global picam2, is_detecting, serial_active, ser
    
    # Signal threads to stop
    is_detecting = False
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
    
    try:
        if ser:
            ser.close()
            os.system('fuser -k /dev/ttyACM0')
    except:
        pass
        
    # You might want to add any additional cleanup code here
    
@app.route('/system_health')
def system_health():
    import psutil
    load = os.getloadavg()
    return jsonify({
        'cpu_percent': psutil.cpu_percent(interval=1),
        'load_1min': load[0],
        'load_5min': load[1],
        'memory': psutil.virtual_memory().percent,
        'threads': threading.active_count(),
        'serial_queue': serial_queue.qsize(),
        'thermal': get_cpu_temp()  # Add this function
    })

def get_cpu_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return float(f.read()) / 1000
    except:
        return None

if __name__ == '__main__':
    try:
        # Set thread priorities (Linux only)
        def set_thread_priority():
            import os
            os.nice(5)  # Reduce priority of main thread
        
        # Initialize serial first
        print("Initializing serial connection...")
        start_serial()
        
        # Initialize camera with lower priority
        set_thread_priority()
        print("Initializing camera...")
        initialize_camera()
        
        # Start worker threads
        frame_thread = threading.Thread(
            target=get_frame,
            daemon=True,
            name="FrameThread"
        )
        frame_thread.start()
        
        detection_thread = threading.Thread(
            target=detection_worker,
            daemon=True,
            name="DetectionThread"
        )
        detection_thread.start()
        
        # Keep main thread at lower priority
        set_thread_priority()
        app.run(host='0.0.0.0', port=5000, threaded=True)
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        cleanup()
		   
