"""Local server for the hand-rating pages. Saves ratings straight into the project.

Every score, tag or note change is written to data/handrate/ratings/handrate_batch_XX.csv,
so nothing depends on browser downloads or browser storage.

    .venv/bin/python scripts/rate_server.py
    open http://localhost:8765/pages/batch_01.html
"""

import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from datacurate.io import DATA

ROOT = DATA / "handrate"
OUT = ROOT / "ratings"
PORT = 8765


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        m = re.fullmatch(r"/save/(batch_\d{2})", self.path)
        if not m:
            return self.send_error(404)
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if not body.startswith(b"key,rating,"):
            return self.send_error(400, "not a ratings CSV")
        OUT.mkdir(exist_ok=True)
        path = OUT / f"handrate_{m[1]}.csv"
        tmp = OUT / f".{path.name}.tmp"  # write-then-rename: a crash never leaves half a file
        tmp.write_bytes(body)
        if path.exists():  # keep the previous version as a hidden backup
            path.replace(OUT / f".{path.name}.bak")
        tmp.replace(path)
        rated = sum(1 for line in body.decode().splitlines()[1:] if line.split(",")[1])
        print(f"saved {path.name}: {rated} rated", flush=True)
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):  # keep the terminal quiet except for saves
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), partial(Handler, directory=str(ROOT)))
    print(f"Rating pages: http://localhost:{PORT}/pages/batch_01.html  (Ctrl+C to stop)")
    print(f"Ratings are saved to {OUT}", flush=True)
    server.serve_forever()
