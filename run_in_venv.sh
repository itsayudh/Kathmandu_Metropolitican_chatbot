#!/bin/bash
# Path to your virtual environment
VENV_PATH="/home/ayudh/My_Folder/Ninjainfosys/projects/chatbot_version_english/chatbot_version_english"

# Debug info for logs
echo "Activating virtual environment at: $VENV_PATH" >&2
echo "Args: $@" >&2

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Run the Python script with arguments
python "$@"


#!/bin/bash
# PYTHON_PATH="/home/ayudh/My_Folder/Ninjainfosys/projects/chatbot_version_english/chatbot_version_english/bin/python"

# echo "Running as user: $(whoami)" >&2
# echo "PATH: $PATH" >&2
# echo "Python Path: $PYTHON_PATH" >&2
# echo "Python Version:" >&2
# "$PYTHON_PATH" --version >&2
# echo "Pydantic version:" >&2
# "$PYTHON_PATH" -m pip show pydantic >&2 || echo "Pydantic not found in this Python env" >&2

# exec "$PYTHON_PATH" "$@"

