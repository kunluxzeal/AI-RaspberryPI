#!/bin/bash

# Kill any stray Chromium instances
pkill -f chromium-browser 2>/dev/null

# Start the Flask backend
echo "Starting Flask backend..."
/usr/bin/python3 /home/Jimmy/Documents/TFLITE/IMG_CLASS/yam_img_classification.py &
BACKEND_PID=$!

# Wait for Flask to be ready
echo "Waiting for Flask backend to be ready..."
until curl -s http://127.0.0.1:5000 > /dev/null; do
    sleep 2
done

echo "Flask is up! Launching Chromium..."

# Launch Chromium in kiosk mode
/usr/bin/chromium-browser --noerrdialogs --disable-session-crashed-bubble \
    --disable-infobars --kiosk http://127.0.0.1:5000 \
    --force-device-scale-factor=0.70 &

# Wait on backend
wait $BACKEND_PID

