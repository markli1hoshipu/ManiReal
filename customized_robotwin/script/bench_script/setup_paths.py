#  Add paths to the system path
import sys
from pathlib import Path
import os

def setup_paths():
    robotwin_root = Path(os.environ["ROBOTWIN_ROOT"])
    bench_root = Path(os.environ["BENCH_ROOT"])
    for p in [robotwin_root, bench_root]:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
        # if f"{robotwin_root}/script" not in sys.path:
        #     sys.path.insert(0, f"{robotwin_root}/script")