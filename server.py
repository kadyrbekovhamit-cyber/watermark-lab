"""Single-process loopback-only UI server. Does not persist submissions."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from watermark_lab import analyze_text, inspect_payload

ROOT = Path(__file__).resolve().parent
MAX_BODY = 512_000
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8")}
FIXTURES = {"marked_correct_key", "unmarked", "marked_other_key"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # No submitted content, profile keys, or request paths in logs.

    def send(self, code, body, kind="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def local_request(self):
        port = self.server.server_port
        allowed_hosts = {"127.0.0.1:%s" % port, "localhost:%s" % port}
        if self.headers.get("Host") not in allowed_hosts:
            self.send(403, {"error": "Loopback Host required"})
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + host for host in allowed_hosts}:
            self.send(403, {"error": "Cross-origin requests are not allowed"})
            return False
        return True

    def do_GET(self):
        if not self.local_request():
            return
        path = urlsplit(self.path).path
        if path in ASSETS:
            name, kind = ASSETS[path]
            self.send(200, (ROOT / "web" / name).read_bytes(), kind)
        elif path == "/api/status":
            self.send(200, {"version": "0.1.0", "claude_detector": "unavailable",
                            "external_requests": False, "stores_submissions": False})
        else:
            self.send(404, {"error": "Not found"})

    def do_POST(self):
        if not self.local_request():
            return
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self.send(415, {"error": "application/json required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 <= length <= MAX_BODY:
                self.send(413, {"error": "Request exceeds 512 KB or has no valid length"})
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON object required")
            if self.path == "/api/analyze":
                result = analyze_text(payload.get("text"))
            elif self.path == "/api/tokens":
                result = inspect_payload(payload)
            elif self.path == "/api/demo":
                kind = payload.get("kind")
                if not isinstance(kind, str) or kind not in FIXTURES:
                    raise ValueError("Unknown demo")
                data = json.loads((ROOT / "evidence" / ("demo-" + kind + ".json")).read_text())
                # Score again; a canned verdict is never used as a detector.
                result = inspect_payload(data["payload"])
                result["demo_kind"] = kind
                result["synthetic_example"] = True
            else:
                self.send(404, {"error": "Not found"})
                return
            self.send(200, result)
        except (ValueError, UnicodeError) as error:
            self.send(400, {"error": str(error)})
        except FileNotFoundError:
            self.send(503, {"error": "Demo evidence missing; run validate.py first"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4548)
    args = parser.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print("Watermark Lab: http://127.0.0.1:%s (local only; no media or vendor calls)" % server.server_port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
