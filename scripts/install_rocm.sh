#!/bin/bash
# ROCm Installation Helper for Ubuntu 22.04
set -e

echo "🔧 Installing AMD ROCm..."

# Add ROCm repo
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.0 jammy main" \
  | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt-get update
sudo apt-get install -y rocm-dev rocm-libs hip-base

# Add user to groups
sudo usermod -aG video,render $USER

echo "✅ ROCm installed. Please reboot and re-login."
echo "   Then verify with: rocminfo"
