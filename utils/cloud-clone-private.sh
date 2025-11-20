#!/bin/bash

REPO_URL="..."
FOLDER_LLM="..."
BRANCH_NAME="..." 

ssh-keygen -t ed25519 -C "tpu-vm-key" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
read -r -p "Did I install/add the SSH key to GitHub? Continue? [y/N]: " response
case "$response" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Operation canceled."; exit 1;;
esac
mkdir llm-finetuning
cd llm-finetuning

git init

git remote add origin $REPO_URL

git sparse-checkout init --cone

git sparse-checkout set $FOLDER_LLM

git pull origin $BRANCH_NAME --depth=1

python3 -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices())"

read -r -p "Install required dependecies? [y/N]: " response
case "$response" in
    [yY]|[yY][eE][sS]) 
        pip install --upgrade pip
        pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html
        pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
        pip install transformers
        pip install datasets
        echo "Testing XLA after installation..."
        python3 -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices())"
        ;;
    *) echo "Canceled"; exit 1;;
esac

echo "Setup complete!"