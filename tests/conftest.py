import sys
from pathlib import Path

tests_dir = Path(__file__).parent.resolve()

if str(tests_dir) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(tests_dir))
