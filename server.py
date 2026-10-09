"""Web panel for OryvexAI: a tiny threaded HTTP server using only the standard library."""
from __future__ import annotations

import hmac
import json
import threading
import uuid
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from engine import BOT_NAME, ChatSession, OryvexEngine
from wiki_trainer import WikiTrainer

MAX_BODY_BYTES = 1_000_000
MAX_WEB_SESSIONS = 100
INDEX_HTML = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")


class SessionStore:
    def __init__(self, limit: int = MAX_WEB_SESSIONS):
        self._data: "OrderedDict[str, ChatSession]" = OrderedDict()
        self._lock = threading.Lock()
        self._limit = limit

    def get(self, sid: str) -> ChatSession:
        with self._lock:
            if sid not in self._data:
                self._data[sid] = ChatSession()
                while len(self._data) > self._limit:
                    self._data.popitem(last=False)
            self._data.move_to_end(sid)
            return self._data[sid]


def clean_sid(value) -> str:
    sid = str(value or "").strip()[:64]
    return sid if sid and all(ch.isalnum() or ch in "-_" for ch in sid) else uuid.uuid4().hex


class WebHandler(BaseHTTPRequestHandler):
    engine: OryvexEngine = None  # set in run_web
    learner = None               # LearnManager when learn mode is enabled
    trainer: WikiTrainer = None  # Wikipedia Training page (always available)
    admin_token = None
    sessions = SessionStore()
    server_version = f"{BOT_NAME}/1.0"

    def log_message(self, fmt, *args):  # keep the terminal clean
        pass

    # -- helpers
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _admin_ok(self) -> bool:
        given = self.headers.get("X-Admin-Token", "")
        return bool(self.admin_token) and hmac.compare_digest(self.admin_token, given)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("Invalid request size.")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    def _train_stream(self, query: dict):
        """Server-Sent Events: pushes every new log line the moment it is written."""
        try:
            last = int(query.get("since", ["0"])[0])
        except ValueError:
            last = 0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            self.wfile.write(b"retry: 2000\n\n")
            while True:
                entries, state = self.trainer.wait_logs(last, 15.0)
                if entries:
                    last = entries[-1]["id"]
                    for e in entries:
                        self.wfile.write(f"id: {e['id']}\nevent: log\ndata: {json.dumps(e)}\n\n".encode("utf-8"))
                    self.wfile.write(f"event: state\ndata: {json.dumps(self.trainer.status())}\n\n".encode("utf-8"))
                else:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return

    # -- routes
    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif url.path == "/api/info":
            self._json(200, {**self.engine.info(), "learn": self.learner is not None})
        elif url.path == "/api/learn/status":
            if self.learner is None:
                return self._json(404, {"error": "Learn mode is off. Start with: python app.py --web --learn"})
            if not self._admin_ok():
                return self._json(403, {"error": "Bad or missing admin token."})
            self._json(200, self.learner.status())
        elif url.path == "/api/train/status":
            try:
                since = int(parse_qs(url.query)["since"][0])
            except (KeyError, ValueError, IndexError):
                since = None
            self._json(200, self.trainer.status(since))
        elif url.path == "/api/train/stream":
            self._train_stream(parse_qs(url.query))
        elif url.path == "/api/train/download":
            if not self.trainer.records:
                return self._json(404, {"error": "Nothing to download yet. Run a training first."})
            body = self.trainer.chat_jsonl().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="chat.jsonl"')
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/history":
            sid = clean_sid(parse_qs(url.query).get("session", [""])[0])
            self._json(200, {"messages": self.sessions.get(sid).transcript})
        elif url.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        url = urlparse(self.path)
        try:
            data = self._read_json()
        except Exception as exc:
            return self._json(400, {"error": str(exc)})

        if url.path.startswith("/api/train/"):
            action = url.path.rsplit("/", 1)[-1]
            if action == "start":
                err = self.trainer.start(data)
            elif action == "stop":
                self.trainer.stop()
                err = None
            elif action == "clear":
                err = self.trainer.clear()
            else:
                return self._json(404, {"error": "Not found"})
            return self._json(400 if err else 200, {"error": err} if err else {"ok": True})

        if url.path.startswith("/api/learn/"):
            if self.learner is None:
                return self._json(404, {"error": "Learn mode is off. Start with: python app.py --web --learn"})
            if not self._admin_ok():
                return self._json(403, {"error": "Bad or missing admin token."})
            action = url.path.rsplit("/", 1)[-1]
            if action == "start":
                err = self.learner.start(data)
                return self._json(400 if err else 200, {"error": err} if err else {"ok": True})
            if action == "stop":
                self.learner.stop()
                return self._json(200, {"ok": True})
            if action == "rollback":
                err = self.learner.rollback()
                return self._json(400 if err else 200, {"error": err} if err else {"ok": True})
            return self._json(404, {"error": "Not found"})

        if url.path == "/api/reset":
            self.sessions.get(clean_sid(data.get("session"))).reset()
            return self._json(200, {"ok": True})

        if url.path != "/api/chat":
            return self._json(404, {"error": "Not found"})

        message = str(data.get("message", "")).strip()
        if not message:
            return self._json(400, {"error": "Empty message."})
        session = self.sessions.get(clean_sid(data.get("session")))

        # Stream plain text chunks; the page reads them as they arrive.
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        cancel = threading.Event()
        gen = self.engine.stream_reply(session, message, cancel)
        try:
            for chunk in gen:
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            cancel.set()  # browser hit Stop or closed the tab
            gen.close()
        except Exception as exc:
            try:
                self.wfile.write(f"\n\n[error] {exc}".encode("utf-8"))
                self.wfile.flush()
            except OSError:
                pass


def run_web(engine: OryvexEngine, host: str, port: int, open_browser: bool,
            learner=None, admin_token: str | None = None):
    WebHandler.engine = engine
    WebHandler.learner = learner
    WebHandler.trainer = WikiTrainer()
    WebHandler.admin_token = admin_token
    server = ThreadingHTTPServer((host, port), WebHandler)
    server.daemon_threads = True
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{shown}:{port}"
    print(f"\n[{BOT_NAME}] Web panel running at {url}   (Ctrl+C to stop)")
    if host in ("0.0.0.0", ""):
        print(f"[{BOT_NAME}] Listening on all interfaces: anyone on your network can use this panel.")
    if learner is not None:
        print(f"[{BOT_NAME}] LEARN MODE is ON. Open the panel with this link (it carries the admin token):")
        print(f"[{BOT_NAME}]   {url}/#admin={admin_token}")
        print(f"[{BOT_NAME}] Keep that token private: it lets whoever has it make the model fetch pages and retrain.")
        url = f"{url}/#admin={admin_token}"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[{BOT_NAME}] Shutting down.")
    finally:
        server.server_close()
