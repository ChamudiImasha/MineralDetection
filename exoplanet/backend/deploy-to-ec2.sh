#!/bin/bash
# EC2 Backend Setup Script
# Run this script ON your EC2 instance after SSH-ing in

set -e

echo "=== CRISM Mineral Detection Backend - EC2 Setup ==="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3.9+
echo "Installing Python 3.9..."
sudo apt-get install -y python3.9 python3.9-venv python3-pip

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# Create app directory
echo "Creating application directory..."
mkdir -p ~/crism-backend
cd ~/crism-backend

# Clone or copy your code here
echo "📌 Now upload your backend code to ~/crism-backend/"
echo "   You can use: scp -r backend/app/* ec2-user@your-ec2-ip:~/crism-backend/"

# Install Python dependencies
echo "Installing Python dependencies..."
cd ~/crism-backend
pip3 install -r requirements.txt
pip3 install -r requirements_api.txt

# Create systemd service
echo "Creating systemd service..."
sudo tee /etc/systemd/system/crism-api.service > /dev/null <<EOF
[Unit]
Description=CRISM Mineral Detection API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/crism-backend
ExecStart=/usr/bin/python3 /home/$USER/crism-backend/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
echo "Enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable crism-api
sudo systemctl start crism-api

echo ""
echo "✅ Setup complete!"
echo "Backend is running on port 8000"
echo "Configure security group to allow inbound traffic on port 8000"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status crism-api  # Check status"
echo "  sudo systemctl restart crism-api # Restart service"
echo "  sudo journalctl -u crism-api -f  # View logs"
