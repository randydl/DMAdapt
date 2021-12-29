import os
import sys
from pathlib import Path


dirs = [
    'domaintransfer',
    'transferlearning',
]

root = Path(__file__).parents[2]
dirs = [str(root/f'{x}') for x in dirs]
dirs = [p for p in dirs if p not in sys.path]
sys.path.extend(dirs)

root = Path(__file__).parents[1]
os.chdir(root)
