# SPE-GTN

## 项目简介

本项目基于蛋白质序列和图结构信息，构建蛋白质图并使用 Graph Transformer Network 进行蛋白质功能预测。
支持从 PDB 或 FASTA 文件生成图数据，训练、测试和预测模型。

## 项目目录

```
P-GTNS/
│
├─ data/processed/        # 处理后的序列或结构文件
├─ graph_embding.py       # 图构建及相关函数
├─ train.py               # 训练脚本
├─ test.py                # 测试脚本
├─ predictor.py           # 预测脚本
└─ requirements.txt       # 依赖库
```

## 安装依赖

首先安装 Python 依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

将蛋白质序列或 PDB 文件放入 `data/processed/` 文件夹。
GO 注释文件请放置在 `data/` 目录下，例如：

```
data/nrPDB-GO_2024.06.24_annot.tsv
```

## 训练模型

使用 `train.py` 训练模型并生成权重：

```bash
python train.py
```

## 测试模型

使用 `test.py` 对训练好的模型进行测试：

```bash
python test.py
```

## 进行预测

使用 `predictor.py` 对新蛋白质数据进行功能预测：

```bash
python predictor.py
```

## 注意事项

* 输入文件格式必须正确：

  * PDB 文件用于三维结构图生成
  * FASTA 文件仅能生成序列邻接图
* GPU 建议使用 CUDA 支持，否则可改用 CPU 训练
* ESM 模型权重需放置在 `model_weight/esm1b.pt`

