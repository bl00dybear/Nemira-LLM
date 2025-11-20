#!/bin/bash

REPO_URL="git@github.com:bl00dybear/Nemira-LLM.git"
FOLDER_LLM="llm-finetuning"
BRANCH_NAME="main" 

mkdir -p llm-finetuning
cd llm-finetuning

git init
git remote add origin $REPO_URL
git sparse-checkout init --cone
git sparse-checkout set $FOLDER_LLM

git pull origin $BRANCH_NAME --depth=1

python3 -c "import torch_xla.core.xla_model as xm; print('Dispozitive XLA detectate:', xm.get_xla_supported_devices())" 2>/dev/null || echo "XLA nu este încă instalat sau configurat."

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
    *) echo "Canceled.";;
esac

echo "Setup complete!"