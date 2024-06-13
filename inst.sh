#!/bin/bash

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip && \
python -m pip install torch==2.1.1 torchvision==0.16.1 \
      --index-url https://download.pytorch.org/whl/cu121 && \
python -m pip install \
            torchmetrics==1.2.0 \
            lion-pytorch==0.1.2 \
            transformers==4.34.1 \
            datasets==2.14.6 \
            lightning==2.1.0 \
            humanize==4.8.0 \
            einops==0.7.0 \
            wandb==0.15.12 \
            qiniu==7.12.0 \
            rich==13.6.0 \
            colorful==0.5.5 \
            jsonlines==4.0.0 \
            -i https://pypi.tuna.tsinghua.edu.cn/simple

cd abseil-py && python setup.py install && cd ..
# git clone https://github.com/abseil/abseil-py.git && cd abseil-py && python setup.py install && cd ..
