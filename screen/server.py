import base64
import os
import subprocess
import tempfile

from flask import Flask, jsonify, request

app = Flask(__name__)

DISPLAY = os.environ.get("DISPLAY", ":99")
_chromium_proc = None


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/open", methods=["POST"])
def open_url():
    global _chromium_proc
    data = request.get_json(force=True)
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "url required"}), 400

    if _chromium_proc and _chromium_proc.poll() is None:
        _chromium_proc.terminate()

    _chromium_proc = subprocess.Popen(
        [
            "chromium",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-software-rasterizer",
            url,
        ],
        env={**os.environ, "DISPLAY": DISPLAY},
    )
    return jsonify({"status": "opened", "url": url})


@app.route("/screenshot")
def screenshot():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name

    result = subprocess.run(
        ["import", "-display", DISPLAY, "-window", "root", path],
        capture_output=True,
    )
    if result.returncode != 0:
        return jsonify({"error": result.stderr.decode()}), 500

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    os.unlink(path)
    return jsonify({"image": data, "format": "png"})


@app.route("/click", methods=["POST"])
def click():
    data = request.get_json(force=True)
    x = data.get("x")
    y = data.get("y")
    if x is None or y is None:
        return jsonify({"error": "x and y required"}), 400

    subprocess.run(
        ["xdotool", "mousemove", str(x), str(y), "click", "1"],
        env={**os.environ, "DISPLAY": DISPLAY},
    )
    return jsonify({"status": "clicked", "x": x, "y": y})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
