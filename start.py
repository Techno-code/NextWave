"""Start the Quant Insight with one command:  python start.py
It checks Python, installs what is missing, builds the problem-catalogue index if it finds the two CSV files,
starts the app on a free port and opens your browser. Options: --no-browser, --port N, --check (prepare only, do not serve)."""
import argparse, importlib.util, os, socket, subprocess, sys, threading, time, webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NEED = {"fastapi": "fastapi", "uvicorn": "uvicorn", "multipart": "python-multipart", "pypdf": "pypdf", "docx": "python-docx", "openpyxl": "openpyxl", "numpy": "numpy", "sklearn": "scikit-learn", "joblib": "joblib", "scipy": "scipy"}
P_NAME, S_NAME = "business_problems_1M.csv", "business_problem_solutions_1M.csv"

def say(m): print(m, flush=True)
def fail(m): say("\nCannot start: " + m); sys.exit(1)

def ensure_packages():
    missing = [pip for mod, pip in NEED.items() if importlib.util.find_spec(mod) is None]
    if not missing: return
    say("Installing what is missing: " + ", ".join(missing) + " (one time, needs internet) ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing])
    if r.returncode:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", *missing])
    if r.returncode: fail("the packages could not be installed. Run:  pip install -r requirements.txt  and then start again.")

def find_csvs():
    homes = [ROOT, ROOT / "data", ROOT / "catalogue", Path.cwd(), Path.home() / "Downloads"]
    for d in homes:
        if (d / P_NAME).exists() and (d / S_NAME).exists(): return d / P_NAME, d / S_NAME
    return None

def ensure_catalogue():
    from app import bigdata
    if bigdata.available(): say("Problem catalogue: ready."); return
    found = find_csvs()
    if not found:
        say(f"Problem catalogue: not built. To enable catalogue search, put {P_NAME} and {S_NAME} in the 'catalogue' folder and start again. Everything else works without it."); return
    say("Building the problem catalogue index (about 20 seconds, one time) ...")
    bigdata.build(*found, log=lambda m: None); say("Problem catalogue: ready.")

def free_port(want):
    for p in range(want, want + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0: return p
    fail("no free port was found between %d and %d." % (want, want + 19))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-browser", action="store_true"); ap.add_argument("--port", type=int, default=8000); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    if sys.version_info < (3, 10): fail(f"Python 3.10 or newer is needed; this is {sys.version.split()[0]}. Install a newer Python from python.org.")
    os.chdir(ROOT); sys.path.insert(0, str(ROOT)); ensure_packages(); ensure_catalogue()
    if a.check: say("Check complete. Run  python start.py  to start the app."); return
    port = free_port(a.port); url = f"http://127.0.0.1:{port}"
    say(f"\nQuant Insight is starting at {url}\nKeep this window open while you use it. Press Ctrl+C to stop.\n")
    if not a.no_browser: threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    import uvicorn; uvicorn.run("app.main:app", host="127.0.0.1", port=port, log_level="warning")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: say("\nStopped.")
