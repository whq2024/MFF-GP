<h1 align="center">
Multimodal Dynamic Fusion Framework: Multilevel Feature Fusion Guided by Prompts
</h1>

<h4 align="center">
  Lei Pan <sup>&dagger;</sup> &nbsp; 
  <a href="https://github.com/whq2024/">HuanQing Wu</a> &nbsp;
</h4>

<br>

This is the repo for the official implementation of the paper: [Multimodal Dynamic Fusion Framework: Multilevel Feature Fusion Guided by Prompts](https://onlinelibrary.wiley.com/doi/full/10.1111/exsy.13668)

## Introduction

In this study, we investigate the impact of hierarchical feature fusion with pre-training models on multimodal fusion tasks. We utilize a dynamic neural network approach and several kinds of fusion strategies to fuse features at different levels, enabling each layer's intermediate feature representation to select the fusion strategy that best satisfies the requirements of the task. The method gives excellent results with a small number of parameter updates. It can be used as a novel analytical tool for modal fusion based on intermediate representations of pre-training models. Following performing a variety of experiments, we observed that using appropriate fusion techniques at different levels is particularly beneficial for modal fusion tasks.

## Method

<div align=center>
	<img src="res/model-arch.png" width="800" >
</div>

This study introduces MFF-GP (Multilevel Feature Fusion Guided by Prompts), a novel method guiding Dynamic Neural Networks (DNN) to regulate the fusion of hierarchical features and also allow prompt vectors to fine-tune unimodal pre-training models. Specifically, by con- catenating prompt vectors with modality-specific raw features as inputs to the pre-training model, it becomes possible to fine-tune the pre-training model while guiding a dynamic neural network to dynamically route for selecting an appropriate fusion module to handle features at each hierarchy. This approach offers a more appropriate fusion method by considering hierarchical feature representations for different tasks. Furthermore, MFF-GP supports plug-and-play for fusion modules, enabling the addition or removal of fusion modules conve- niently. This significantly enhances the flexibility of fusion, offering adaptability for several multimodal tasks.

## Project Structure

```txt
.
├── README.md: a description document of this project.
├── abseil-py: a library for building Python applications and logging.
├── configs: config files for various datasests and running settings.
├── data: utils for loading dataset data.
├── inst.sh: a shell script for linux system to install dependent env.
├── main.py: the program operation entry file.
├── models: the modules of all model.
├── model-arch.png: The overall arch diagram of the MFF-GP.
└── utils: common utils for the program.
```

## Dataset Format

The default dataset is stored in the `datasets` directory in the root directory, which is structured as follows:

```txt
datasets
├── mmimdb
│   ├── dataset
│   │   ├── 0070505.jpeg
│   │   ├── 0070505.json
│   │   ├── 0093582.jpeg
│   │   ├── 0093582.json
│   │   └── ....
│   └── split.json
├── SNLI-VE
│   ├── images
│   │   ├── 3373544964.jpg
│   │   ├── 556601134.jpg
│   │   └── ....
│   ├── snli_ve_dev.jsonl
│   ├── snli_ve_test.jsonl
│   └── snli_ve_train.jsonl
└── UPMC-Food101
    ├── images
    │   ├── test
    │   └── train
    └── texts
        ├── test_titles.csv
        └── train_titles.csv
```

> Downloading datasets: [MM-IMDB](https://archive.org/details/mmimdb), [SNLI-VE](https://github.com/necla-ml/SNLI-VE) and [UPMC-Food101](https://www.kaggle.com/datasets/gianmarco96/upmcfood101)

## Environment

Required packages and dependencies are listed in **inst.sh**. You can use the conda installation environment with the following command:

```bash

# create env by conda
conda create -n mff-gp python=3.11
conda activate mff-gp

# for Linux
chmod +x inst.sh
./inst.sh

# for Windows
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

# for update abseil-py
# git clone https://github.com/abseil/abseil-py.git && cd abseil-py && python setup.py install && cd ..

```

> Please note that you need to create a [wandb account](https://wandb.ai/) first, as all our log information is tracked and managed by wandb, when training with `--use_wandb` param.

## Training and Evaluation

Launch the `main.py` script with the following command:

```bash
# training and testing for the MM-IMDB dataset
python main.py -c configs/mmimdb_4x24G.yml --checkpoint_monitor f1_micro --patience 10 --use_wandb

# training and testing for the SNLI-VE dataset
python main.py -c configs/snli_ve_8G.yml --use_wandb

# only training for the UPMC-Food101 dataset
python main.py -c configs/food101_8G.yml --type fit --use_wandb

# only testing for the UPMC-Food101 dataset with experiment_id and weight_name(default 'last.ckpt')
python main.py -c configs/food101_8G.yml --type test --experiment_id <EXPERTMENT_ID_VAL> --weight_name <WEIGHT_NAME_VAL>
```

> NOTE: For more information on the parameters, see the `main.py` file and modify the hyperparameters via the configuration file `*.yml`. (For example, the unmask prompt vectors can modify the 'masked_prompt' parameter in `*.yml` files.)

<!-- ## Citation

Please consider citing our work if you find our projects useful in your research:

```biber
...
```
 -->

## Acknowledgments

Our code is heavily based on projects like [Huggingface transformers](https://huggingface.co/docs/transformers/index), [lightning](https://lightning.ai/docs/pytorch/stable/) and [wandb](https://wandb.ai/). Thanks for their splendid work!
