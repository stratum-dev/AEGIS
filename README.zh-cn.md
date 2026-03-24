# AEGIS 深度学习代码漏洞检测模型

在开始本项目之前，你需要熟悉 Huggingface 平台及其 API 的操作。

## 配置

本项目在以下环境和配置中训练，读者可作为参考，根据本机配置调整适当模型训练和推理参数。

- NVIDIA RTX 5880, 48GB
- CUDA: 12.4
- Memory 128GB
- Intel(R) Core(TM) i9-14900K x86-64, 24 Cores * 2 threads.
- Ubuntu 22.04, Python 3.11
- 除此之外，至少还需要 20 GB 的空闲空间用于储存数据集、模型。

为避免兼容性问题，请使用类 Unix 操作系统（如 MacOS, Linux, WSL 等）运行项目。

## 安装

载入 Conda 环境并安装依赖
```bash
conda env create --name aegis -f environment.yml
```

激活当前 Conda 环境

```bash
conda activate aegis
```

如果你在中国大陆地区，需要确保你的机器能正常访问到 HuggingFace。

你可以使用 HuggingFace 的中国镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后下载预训练 CweBERT 模型和本项目的数据集。

```bash
pip install -U huggingface_hub
huggingface-cli download --resume-download codemetic/CweBERT-mlm
huggingface-cli download --repo-type dataset codemetic/AEGIS
```

## 预训练

在上一步中，已经下载好了 CweBERT-mlm。预训练 CweBERT-mlm 不是本项研究的主要内容。**所以通常情况下，你不需要预训练 CweBERT-mlm。**

可跳到[正式训练](#正式训练)这一步。

如果需要预训练，请按照以下步骤执行。

首先，预训练分词器：

```bash
python pretokenize.py
```

再执行预训练
```bash
python mlm.py
```

可评估预训练脚本
```bash
python mlm-eval.py
```

## 正式训练

在指定数据集上训练 CweBERT。

```bash
python train.py
```

训练时，会在根目录下生成类似 `output_xxxx_20xxxxxx-xx-xx-xx` 的输出目录，里面会储存 Checkpoint。评估结果也会输出到该目录中。

这里的 Checkpoint 是按最优 pareto 前沿对应的 epoch 保存的。

再直接评估训练好的模型。评估时，你可以指定要评估的 checkpoint。

```bash
python eval.py \
  --model_dir output_megavul_20260111-17-43-19 \
  --subset_name megavul \
  --checkpoint 15 \
  --batch_size 32 \
  --random_seed 42
```

然后会在 `output_xxxx_20xxxxxx-xx-xx-xx/evaluation-x` 的目录下输出评估结果。

## 以本项目为基线实验

如果需要以本项目或使用自定义数据集为基线实验，请按照以下步骤执行：

+ 注册 HuggingFace 账号
+ 自定义数据集应当满足以下条件：
  - 使用 Parquet 格式，具备 id，label，cwe 和 source 列。label 为 Boolean 类型，其余均为 string 类型。
  - split 的训练、测试和验证集名称为 val,test和train。
+ 将数据集上传至 Huggingface。
+ 将训练配置的 SUBSET_NAME 、DATASET_NAME 设置为你的名称。 