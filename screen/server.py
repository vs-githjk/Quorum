"""
screen/server.py — Quorum Screen API

Controls a Playwright browser running on Xvfb.

Threading model: Playwright's sync API is greenlet-based and must run on a
single dedicated thread. All Playwright operations are dispatched via a work
queue (_pw_queue) to that thread. Flask request threads submit work and block
on a Future until the result is ready.

Endpoints:
    GET  /health
    POST /open            {"url": str}           navigate (backwards compat)
    POST /navigate        {"url": str}           navigate
    GET  /screenshot      base64 PNG of current page
    POST /fill            {"selector": str, "text": str}
    POST /click           {"x": int, "y": int}
    POST /render_html     {"html": str}
    POST /act             {"instruction": str}   SSE vision loop
    POST /act/cancel      cancel running loop
"""

import base64
import concurrent.futures as cf
import json
import logging
import os
import queue
import re
import tempfile
import threading
import time

import requests as req_lib
from flask import Flask, Response, jsonify, request
from playwright.sync_api import sync_playwright

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

DISPLAY          = os.environ.get("DISPLAY", ":99")
OPENROUTER_KEY   = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("VISION_MODEL", os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o"))
OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"

# ── Playwright thread + queue ─────────────────────────────────────────────────

_pw_queue: queue.Queue = queue.Queue()
_page = None


def _pw_worker():
    """Single dedicated thread that owns all Playwright state."""
    global _page
    logger.info("Playwright thread starting (DISPLAY=%s)", DISPLAY)
    pw = sync_playwright().start()
    ctx = pw.chromium.launch_persistent_context(
        user_data_dir="/tmp/chromium-profile",
        headless=False,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-software-rasterizer",
        ],
        viewport={"width": 1280, "height": 720},
        env={**os.environ, "DISPLAY": DISPLAY},
    )
    _page = ctx.new_page()
    _page.goto("about:blank")
    logger.info("Playwright ready")

    while True:
        fn, fut = _pw_queue.get()
        if fn is None:
            break
        try:
            result = fn()
            try:
                fut.set_result(result)
            except cf.InvalidStateError:
                pass  # caller already timed out — ignore
        except Exception as exc:
            try:
                fut.set_exception(exc)
            except cf.InvalidStateError:
                pass  # caller already timed out — ignore


def _pw(fn, timeout: float = 30.0):
    """Dispatch fn to the Playwright thread, block until done, return result."""
    fut: cf.Future = cf.Future()
    _pw_queue.put((fn, fut))
    return fut.result(timeout=timeout)


# ── Vision loop state ─────────────────────────────────────────────────────────

_cancel_requested = False
_current_loop_id = None

# ── Vision loop system prompt ─────────────────────────────────────────────────

VISION_SYSTEM_PROMPT = """\
You are a browser automation agent controlling a real browser. You see a screenshot and must decide the single next action to complete the given task.

RESPOND WITH ONLY A SINGLE JSON OBJECT. NO MARKDOWN FENCES. NO EXPLANATION. JUST JSON.

Action formats:
{"action": "navigate", "url": "https://...", "description": "Navigating to ..."}
{"action": "click", "selector": "CSS selector", "description": "Clicking ..."}
{"action": "click_text", "text": "visible text", "description": "Clicking ..."}
{"action": "type", "selector": "CSS selector", "text": "text to type", "description": "Typing ..."}
{"action": "key", "key": "Enter", "description": "Pressing Enter"}
{"action": "scroll", "direction": "down", "description": "Scrolling down"}
{"action": "wait", "seconds": 2, "description": "Waiting for page to load"}
{"action": "done", "summary": "What was accomplished"}
{"action": "error", "summary": "Why the task cannot be completed"}

Rules:
- Prefer specific CSS selectors (id, name, aria-label) over generic ones
- Use "click_text" when you can only identify an element by its visible label
- After typing into a search field, always follow with {"action": "key", "key": "Enter"}
- Use "wait" when a page appears to still be loading
- Use "done" ONLY when the task is fully complete and visible on screen — not after just navigating or typing
- If you just navigated or typed but the result isn't visible yet, use "wait" or continue with the next action
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_action(content: str) -> dict:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    return json.loads(content.strip())


def _execute_action(action_data: dict) -> None:
    """Must be called from inside _pw() — runs on the Playwright thread."""
    action = action_data.get("action", "")

    if action == "navigate":
        _page.goto(action_data["url"], wait_until="domcontentloaded", timeout=15000)

    elif action == "click":
        try:
            _page.click(action_data["selector"], timeout=5000)
        except Exception:
            _page.click(action_data["selector"], force=True, timeout=3000)

    elif action == "click_text":
        _page.get_by_text(action_data["text"]).first.click(timeout=5000)

    elif action == "type":
        _page.fill(action_data["selector"], action_data["text"], timeout=5000)

    elif action == "key":
        _page.keyboard.press(action_data["key"])

    elif action == "scroll":
        amount = 500 if action_data.get("direction", "down") == "down" else -500
        _page.evaluate(f"window.scrollBy(0, {amount})")

    elif action == "wait":
        time.sleep(min(float(action_data.get("seconds", 2)), 10))


# ── Vision loop ───────────────────────────────────────────────────────────────

def _vision_loop_generator(instruction: str, loop_id: str):
    """SSE generator — screenshot → LLM → action, streamed to the caller."""
    global _cancel_requested

    history = []
    step = 0

    try:
        while True:
            if _cancel_requested and _current_loop_id == loop_id:
                yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                return

            # Screenshot (Playwright thread)
            try:
                screenshot_bytes = _pw(lambda: _page.screenshot(full_page=False))
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Screenshot failed: {e}'})}\n\n"
                return

            # Call OpenRouter (plain HTTP — no Playwright involvement)
            messages = [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                *history[-6:],
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Task: {instruction}\nStep {step + 1}: What is the next action?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                        },
                    ],
                },
            ]
            try:
                resp = req_lib.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={"model": OPENROUTER_MODEL, "messages": messages},
                    timeout=30,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'LLM call failed: {e}'})}\n\n"
                return

            # Parse
            try:
                action_data = _parse_action(content)
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Bad action JSON: {e} — raw: {content[:120]}'})}\n\n"
                return

            action_type = action_data.get("action", "")
            description = action_data.get("description", action_type)

            yield f"data: {json.dumps({'type': 'step', 'action': action_type, 'description': description, 'step': step + 1})}\n\n"

            if action_type in ("done", "error"):
                summary = action_data.get("summary", description)
                yield f"data: {json.dumps({'type': 'done', 'summary': summary, 'steps': step + 1})}\n\n"
                return

            # Execute (Playwright thread)
            try:
                captured = dict(action_data)  # capture for lambda
                _pw(lambda: _execute_action(captured), timeout=20.0)
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Action failed: {e}'})}\n\n"
                return

            history.append({"role": "assistant", "content": content})
            history.append({
                "role": "user",
                "content": f"Executed '{action_type}': {description}",
            })

            step += 1
            time.sleep(0.5)

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/open", methods=["POST"])
@app.route("/navigate", methods=["POST"])
def navigate():
    data = request.get_json(force=True)
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "url required"}), 400
    _pw(lambda: _page.goto(url, wait_until="domcontentloaded", timeout=15000))
    return jsonify({"status": "opened", "url": url})


@app.route("/screenshot")
def screenshot():
    try:
        png = _pw(lambda: _page.screenshot(full_page=False))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"image": base64.b64encode(png).decode(), "format": "png"})


@app.route("/fill", methods=["POST"])
def fill():
    data = request.get_json(force=True)
    selector = data.get("selector", "")
    text = data.get("text", "")
    if not selector:
        return jsonify({"error": "selector required"}), 400
    _pw(lambda: _page.fill(selector, text, timeout=5000))
    return jsonify({"status": "ok"})


@app.route("/click", methods=["POST"])
def click():
    data = request.get_json(force=True)
    x, y = data.get("x"), data.get("y")
    if x is None or y is None:
        return jsonify({"error": "x and y required"}), 400
    _pw(lambda: _page.mouse.click(x, y))
    return jsonify({"status": "clicked", "x": x, "y": y})


@app.route("/render_html", methods=["POST"])
def render_html():
    data = request.get_json(force=True)
    html = data.get("html", "")
    if not html:
        return jsonify({"error": "html required"}), 400
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        f.write(html)
        path = f.name
    _pw(lambda: _page.goto(f"file://{path}", wait_until="domcontentloaded", timeout=10000))
    return jsonify({"status": "rendered"})


@app.route("/act", methods=["POST"])
def act():
    global _cancel_requested, _current_loop_id
    data = request.get_json(force=True)
    instruction = data.get("instruction", "").strip()
    if not instruction:
        return jsonify({"error": "instruction required"}), 400
    if not OPENROUTER_KEY:
        return jsonify({"error": "OPENROUTER_API_KEY not set in container"}), 500

    import uuid
    loop_id = str(uuid.uuid4())
    _current_loop_id = loop_id
    _cancel_requested = False

    return Response(
        _vision_loop_generator(instruction, loop_id),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/act/cancel", methods=["POST"])
def act_cancel():
    global _cancel_requested
    _cancel_requested = True
    return jsonify({"status": "cancelling"})


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = threading.Thread(target=_pw_worker, daemon=True)
    t.start()
    # Wait for Playwright to be ready
    while _page is None:
        time.sleep(0.1)
    app.run(host="0.0.0.0", port=5000, threaded=True)
