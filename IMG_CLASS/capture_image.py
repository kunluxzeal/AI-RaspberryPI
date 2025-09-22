from picamera2 import Picamera2
import time


# Initialize the camera
picam2 = Picamera2() # default is index 0

#initializre the camera
config = picam2.create_still_configuration(main={"size" :(640, 480)})
picam2. configure(config)

#start camera 
picam2.start()

#wait for the ccamera to warm up
time.sleep(2)

#capture an image
picam2.capture_file("cup.jpg")
print("Image capture and saved as 'cup.jpg'")

#stop the camera
picam2.stop()