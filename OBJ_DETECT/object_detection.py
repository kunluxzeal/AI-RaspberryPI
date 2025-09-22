import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import tflite_runtime.interpreter as tflite

model_path = "./obj_model/ei-object-detection-object-detection-tensorflow-lite-int8-quantized-model.3.lite"
labels = ['battery', 'cup', 'headset']


# Load the TFLite model
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_dtype = input_details[0]['dtype']
print(input_dtype)

img_path = "./images/test-obj/image3.jpg"
orig_img = Image.open(img_path)
# Display the image
plt.figure(figsize=(6, 6))
plt.imshow(orig_img)
plt.title("Original Image")
plt.show()


scale, zero_point = input_details[0]['quantization']
img = orig_img.resize((input_details[0]['shape'][1],
                  input_details[0]['shape'][2]))

img_array = np.array(img, dtype=np.float32) / 255.0
img_array = (img_array / scale + zero_point).clip(-128, 127).astype(np.int8)
input_data = np.expand_dims(img_array, axis=0)

input_data.shape, input_data.dtype

# Inference on Raspi-Zero
start_time = time.time()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
end_time = time.time()
inference_time = (end_time - start_time) * 1000 # Convert to milliseconds
print ("Inference time: {:.1f}ms".format(inference_time))


boxes = interpreter.get_tensor(output_details[1]['index'])[0]
classes = interpreter.get_tensor(output_details[3]['index'])[0]
scores = interpreter.get_tensor(output_details[0]['index'])[0]
num_detections = int(interpreter.get_tensor(output_details[2]['index'])[0])


for i in range(num_detections):
	if scores[i] > 0.5: # Confidence threshold
		print(f"Object {i}:")
		print(f" Bounding Box: {boxes[i]}")
		print(f" Confidence: {scores[i]}")
		print(f" Class: {classes[i]}")
	threshold = 0.5
	plt.figure(figsize=(6,6))
	plt.imshow(orig_img)
	for i in range(num_detections):
		if scores[i] > threshold:
			ymin, xmin, ymax, xmax = boxes[i]
			(left, right, top, bottom) = (xmin * orig_img.width, xmax * orig_img.width, ymin * orig_img.height,ymax * orig_img.height)
			rect = plt.Rectangle((left, top), right-left, bottom-top,
			fill=False, color='red', linewidth=2)
			plt.gca().add_patch(rect)
			class_id = int(classes[i])
			class_name = labels[class_id]
			plt.text(left, top-10, f'{class_name}: {scores[i]:.2f}',
			color='red', fontsize=12, backgroundcolor='white')

