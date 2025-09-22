import numpy as np	
from PIL import Image
import matplotlib.pyplot as plt
import time
import tflite_runtime.interpreter as tflite




#defining my lables and path
model_path ="./models/img_class_model/ei-img_class-transfer-learning-tensorflow-lite-int8-quantized-model.3.lite"
img_path = "./dataset/test/cup2.jpg"
labels = ['background', 'cup', 'mouse']


def image_classification(img_path, model_path, labels, top_k_results=3, apply_softmax=False):
		
	# Load the image
	img = Image.open(img_path)
	plt.figure(figsize=(4, 4))
	plt.imshow(img)
	plt.axis('off')	
	plt.show()

	# Load the TFLite model
	interpreter = tflite.Interpreter(model_path=model_path)
	interpreter.allocate_tensors()

	# Get input and output tensors
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()


	# Preprocess
	img = img.resize((input_details[0]['shape'][1],
	input_details[0]['shape'][2]))
	input_dtype = input_details[0]['dtype']

	if input_dtype == np.uint8:
		input_data = np.expand_dims(np.array(img), axis=0)
		
	elif input_dtype == np.int8:
		scale, zero_point = input_details[0]['quantization']
		img_array = np.array(img, dtype=np.float32) / 255.0
		img_array = (img_array / scale + zero_point).clip(-128, 127).astype(np.int8)
		input_data = np.expand_dims(img_array, axis=0)
		
	else: # float32
		input_data = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
		
	# Inference on Raspi-Zero
		
	start_time = time.time()
	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()
	end_time = time.time()
	inference_time = (end_time - start_time) * 1000 # Convert to millisecond

	# Obtain results
	predictions = interpreter.get_tensor(output_details[0]['index'])[0]

	# Get indices of the top k results
	top_k_indices = np.argsort(predictions)[::-1][:top_k_results]

	# Handle output based on type
	output_dtype = output_details[0]['dtype']

	if output_dtype in [np.int8, np.uint8]:
		# Dequantize the output
		scale, zero_point = output_details[0]['quantization']
		predictions = (predictions.astype(np.float32) - zero_point) * scale
		
	if apply_softmax:
		# Apply softmax
		
		exp_preds = np.exp(predictions - np.max(predictions))
		probabilities = exp_preds / np.sum(exp_preds)
	else:
		probabilities = predictions
		
		
	print("\n\t[PREDICTION] [Prob]\n")
	
	for i in range(top_k_results):
		print("\t{:20}: {:.1f}%".format(labels[top_k_indices[i]],probabilities[top_k_indices[i]] * 100))
	print ("\n\tInference time: {:.1f}ms".format(inference_time))
	
	
image_classification(img_path, model_path, labels, top_k_results=3, apply_softmax=False)
