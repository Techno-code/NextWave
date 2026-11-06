"""The launcher's helpers: finding the catalogue files, choosing a port, and refusing old Python with a plain message."""
import socket, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import start

def test_free_port_skips_a_busy_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(); busy = s.getsockname()[1]
    try: assert start.free_port(busy) != busy
    finally: s.close()
def test_find_csvs_needs_both_files(tmp_path, monkeypatch):
    monkeypatch.setattr(start, "ROOT", tmp_path); monkeypatch.chdir(tmp_path); monkeypatch.setattr(Path, "home", lambda: tmp_path / "nohome")
    assert start.find_csvs() is None
    (tmp_path / "catalogue").mkdir(); (tmp_path / "catalogue" / start.P_NAME).write_text("x"); assert start.find_csvs() is None
    (tmp_path / "catalogue" / start.S_NAME).write_text("y"); p, s = start.find_csvs(); assert p.name == start.P_NAME and s.name == start.S_NAME
def test_every_required_package_is_listed_in_requirements():
    req = (Path(start.ROOT) / "requirements.txt").read_text().lower()
    for pip in start.NEED.values(): assert pip.lower() in req, pip
def test_old_python_gets_a_plain_message(monkeypatch, capsys):
    monkeypatch.setattr(sys, "version_info", (3, 8, 0)); monkeypatch.setattr(sys, "argv", ["start.py", "--check"])
    with pytest.raises(SystemExit): start.main()
    assert "Python 3.10 or newer" in capsys.readouterr().out
