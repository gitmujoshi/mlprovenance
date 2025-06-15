#!/usr/bin/env python3
import sys
from pathlib import Path
import os

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from safety_features.app.app import app

if __name__ == '__main__':
    app.run(debug=True, port=5000) 