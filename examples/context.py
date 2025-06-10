
from pathlib import Path

try:
    import ddforecast
except ImportError:
    import sys
    sys.path.append('..')
    import ddforecast

print("ddforecast version:", ddforecast.__version__)

tmp_dir = Path(__file__).parent / 'tmp'
tmp_dir.mkdir(exist_ok=True)
