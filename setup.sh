#!/bin/bash

curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.6.0/pget_linux_x86_64" && chmod +x /usr/local/bin/pget
pip install triton==2.2.0
# pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torchsde
wget https://github.com/chengzeyi/stable-fast/releases/download/v1.0.0/stable_fast-1.0.0+torch212cu118-cp310-cp310-manylinux2014_x86_64.whl
pip3 install stable_fast-1.0.0+torch212cu118-cp310-cp310-manylinux2014_x86_64.whl
pip install torchvision==0.16.1 torch==2.1.1
pip install packaging pydantic_core annotated_types structlog regex onnx insightface pytz multidict aiohttp
pip install --upgrade omegaconf
pip install diffusers==0.26.0
pip install lightning_utilities ordered_set kornia_rs torchmetrics
pip install transformers
pip install wandb==0.17.0
pip install xformers==0.0.23 --no-deps
pip install IPython