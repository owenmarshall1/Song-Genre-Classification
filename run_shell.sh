#!/bin/bash

echo "Starting Song Genre Classifier..."

# Check if Python exists
if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Please install Python 3.10+ and try again."
    exit 1
fi

# Check for required packages (torch as representative)
python3 - <<EOF
import sys
try:
    import torch
except ImportError:
    sys.exit(1)
EOF

# If torch is missing, install all requirements
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Run the GUI
python3 gui_predict.py
