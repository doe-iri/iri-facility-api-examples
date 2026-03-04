#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
KERNEL_NAME="iri-examples"
DISPLAY_NAME="Python (iri-examples)"

echo "Checking virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment"
source "$VENV_DIR/bin/activate"

echo " Upgrading pip"
python -m pip install --upgrade pip

echo "Installing required packages"
python -m pip install jupyter ipykernel

echo "Registering Jupyter kernel (if needed)"
if ! jupyter kernelspec list | grep -q "$KERNEL_NAME"; then
  python -m ipykernel install --user \
    --name "$KERNEL_NAME" \
    --display-name "$DISPLAY_NAME"
fi

echo "Starting Jupyter Notebook"
exec jupyter-notebook
