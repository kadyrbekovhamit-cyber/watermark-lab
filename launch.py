"""Start the local Watermark Lab and optionally open its browser interface."""
import argparse
import threading
import webbrowser
from http.server import HTTPServer

from server import Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4548)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        server = HTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        parser.exit(1, "Cannot start local server: %s\nTry --port 4549.\n" % error)
    url = "http://127.0.0.1:%s/" % server.server_port
    print("Watermark Lab: %s\nPress Ctrl+C to stop." % url, flush=True)
    if not args.no_browser:
        opener = threading.Timer(0.3, webbrowser.open, args=(url,))
        opener.daemon = True
        opener.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
