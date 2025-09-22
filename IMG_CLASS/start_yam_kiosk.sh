#!/bin/bash

# Kill any stray Chromium instances
pkill -f chromium-browser 2>/dev/null

# Start the backend (Flask app)
echo "Starting Flask backend..."
/usr/bin/python3 /home/Jimmy/Documents/TFLITE/IMG_CLASS/app.py &
BACKEND_PID=$!

# Wait for Flask to be ready
echo "Waiting for Flask to be ready..."
until curl -s http://127.0.0.1:8000 > /dev/null; do
    sleep 2
done

echo "Flask is up! Launching Chromium..."

# Launch Chromium in kiosk mode
/usr/bin/chromium-browser --noerrdialogs --disable-session-crashed-bubble \
    --disable-infobars --kiosk http://127.0.0.1:8000 &
CHROMIUM_PID=$!

# Wait on both processes
wait $BACKEND_PID $CHROMIUM_PID

