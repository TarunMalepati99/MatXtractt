# python3 -m pip install scikit-optimize numpy matplotlib pandas openpyxl scikit-learn xgboost
#!/bin/bash

# Set virtual environment directory name
VENV_DIR="ml_venv"

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Ensure Python 3.6 or higher is used (scikit-optimize requires at least Python 3.6)
PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
REQUIRED_VERSION="3.6"
if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "Error: Python 3.6 or higher is required. Current version is $PYTHON_VERSION. Please upgrade Python and try again."
    exit 1
fi

# Ensure python3-venv is installed (some systems do not include it by default)
if ! python3 -m ensurepip &> /dev/null; then
    echo "Installing Python venv..."
    sudo apt update && sudo apt install python3-venv -y
fi

# Create the virtual environment if it does not exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment ($VENV_DIR)..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required dependencies
echo "Installing dependencies..."
pip install scikit-optimize numpy matplotlib pandas openpyxl scikit-learn xgboost

# Notify the user about the environment setup
echo "✅ Environment setup complete! Use the following command to activate the virtual environment:"
echo ""
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "To deactivate the virtual environment, run:"
echo ""
echo "    deactivate"
echo ""

# Exit script successfully
exit 0

