#!/bin/bash

echo "Updating the sytem and installing dependencies..."
# Update the system and install dependencies
sudo apt update && sudo apt upgrade -y


echo "Setting up Python virtual environment and installing required packages..."
# Create virtual environment
python3 -m venv venv-master
source venv-master/bin/activate

# Upgrade pip
pip install --upgrade pip --break-system-packages

# Install required libraries
pip install torch tensorflow torchvision matplotlib numpy wandb scikit-learn pandas --break-system-packages

# Optional for logging/debugging or if plotting via CLI
pip install tqdm

echo "✅ Environment setup complete. Activate with: source venv-master/bin/activate"
