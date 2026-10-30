"""Lets scripts and tests import sibling-folder modules and the root settings.py."""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "clean"))
sys.path.insert(0, str(PROJECT_DIR / "sql"))
sys.path.insert(0, str(PROJECT_DIR / "warehouse"))
sys.path.insert(0, str(PROJECT_DIR / "baseline"))
sys.path.insert(0, str(PROJECT_DIR / "model"))
sys.path.insert(0, str(PROJECT_DIR / "report"))
sys.path.insert(0, str(PROJECT_DIR / "site"))
sys.path.insert(0, str(PROJECT_DIR / "ingest"))
