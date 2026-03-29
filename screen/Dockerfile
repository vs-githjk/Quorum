FROM debian:bullseye-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99

RUN apt-get update && apt-get install -y \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    xdotool \
    python3 \
    python3-pip \
    imagemagick \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install flask requests playwright
RUN playwright install --with-deps chromium

COPY server.py /app/server.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 6080 5000

CMD ["/app/start.sh"]
