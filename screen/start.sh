#!/bin/bash
set -e

# Start Xvfb virtual display
Xvfb :99 -screen 0 1280x720x24 &
XVFB_PID=$!

# Wait for display to be ready
until xdpyinfo -display :99 >/dev/null 2>&1; do
    sleep 0.1
done

# Start x11vnc — -wait 10 = ~100fps, -nap = low CPU when idle
x11vnc -display :99 -nopw -listen localhost -xkb -forever -nap -wait 10 &

# Start websockify + noVNC
websockify --web /usr/share/novnc 6080 localhost:5900 &

# Start Flask API
cd /app
python3 server.py
