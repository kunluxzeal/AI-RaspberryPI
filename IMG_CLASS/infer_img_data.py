import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import tflite_runtime.interpreter as tflite


#defining my lables and path
model_path ="./models/img_class_model/ei-img_class-transfer-learning-tensorflow-lite-int8-quantized-model.3.lite"
img_path = "./dataset/test/cup1.jpg"
labels = ['background', 'cup4', 'mouse']

# Load the TFLite model
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_dtype = input_details[0]['dtype']

img = Image.open(img_path)
plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.axis('off')
plt.show()


scale, zero_point = input_details[0]['quantization']
img = img.resize((input_details[0]['shape'][1],
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

# Obtain results and map them to the classes
predictions = interpreter.get_tensor(output_details[0]['index'])[0]

# Get indices of the top k results
top_k_results=3
top_k_indices = np.argsort(predictions)[::-1][:top_k_results]

# Get quantization parameters
scale, zero_point = output_details[0]['quantization']

# Dequantize the output
dequantized_output = (predictions.astype(np.float32) - zero_point) * scale
probabilities = dequantized_output

print("\n\t[PREDICTION] [Prob]\n")
for i in range(top_k_results):
    print("\t{:20}: {:.2f}%".format(labels[top_k_indices[i]], probabilities[top_k_indices[i]] * 100))

print ("Inference time: {:.1f}ms".format(inference_time))
