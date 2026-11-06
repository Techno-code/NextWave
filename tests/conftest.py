"""Each test run gets its own empty database, so runs never depend on leftovers from an earlier run."""
import os, tempfile
_dir = tempfile.mkdtemp(prefix="de-test-")
os.environ["DE_DB"] = os.path.join(_dir, "app.db")
os.environ["DE_ALLOW_HTTP_FEEDS"] = "1"
# Retraining writes new model versions; give the tests a private copy so the project's models/ folder is never touched.
import shutil, pathlib
shutil.copytree(pathlib.Path(__file__).resolve().parent.parent / "models", os.path.join(_dir, "models")); os.environ["DE_MODELS"] = os.path.join(_dir, "models")
os.environ["DE_ADMINS"] = "reviewer@example.com"
