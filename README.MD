# AEGIS

**A**daptive **E**mbedding **G**eometry for **I**mbalanced **S**oftware Vulnerability Detection.

---

## Install and Configure

Before starting this project, you need to be familiar with the Huggingface platform and its API.

This project is trained in the following environment and configuration. Readers can use this as a reference and adjust the appropriate model training and inference parameters according to their local configuration.

- NVIDIA RTX 5880, 48GB
- CUDA: 12.4
- Memory 128GB
- Intel(R) Core(TM) i9-14900K x86-64, 24 Cores * 2 threads.
- Ubuntu 22.04, Python 3.11
- In addition, at least 20 GB of free space is required for storing the dataset and model.

Create the conda environment.

```bash
conda env create --name aegis -f environment.yml \
conda activate aegis 
```

And install dependencies.

```bash
pip install -r requirements-pip.txt
```

All datasets are hosted on [Huggingface](https://huggingface.co/datasets/codemetic/AEGIS-dataset). The training program will automatically download You not need to download it manually.

## Training

Edit the [train.py](./train.py) with the configuration, hyperparameters. Then execute

```bash
python train.py
```

## Evaluation

Run the eval.py and follow the instruction from CLI, select the model directory and the checkpoint.

```bash
python eval.py
```
