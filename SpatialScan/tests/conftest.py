import sys
from pathlib import Path

# Allow running the tests without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
