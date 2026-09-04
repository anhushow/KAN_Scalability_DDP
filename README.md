# Scalability Analysis of Distributed Kolmogorov-Arnold Network Training on High-Performance Computing Systems

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![DDP Distributed](https://img.shields.io/badge/Distributed-PyTorch%20DDP-orange.svg)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
[![HPC SLURM](https://img.shields.io/badge/HPC-SLURM%20Batch-green.svg)](https://slurm.schedmd.com/)

[ **English** ](#-english-version) | [ **中文说明** ](#-中文版本)

---

# 🌐 English Version

> **Official Source Code Notice**:  
> This repository contains the official source code, automated benchmarking suite, and reproducible experimental artifacts for the research paper:  
> **"Scalability Analysis of Distributed Kolmogorov-Arnold Network Training on High-Performance Computing Systems"**.

---

## 📖 Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Experimental Benchmark Design](#2-experimental-benchmark-design)
3. [Summary of Key Findings](#3-summary-of-key-findings)
4. [Repository Structure](#4-repository-structure)
5. [Environment Setup & Prerequisites](#5-environment-setup--prerequisites)
6. [Quick Start & Reproduction Guide](#6-quick-start--reproduction-guide)
   - [Local & Single-Node Multi-GPU Execution](#61-local--single-node-multi-gpu-execution)
   - [Submitting Individual SLURM Batch Jobs](#62-submitting-individual-slurm-batch-jobs)
   - [Automated Batch Queue & Monitoring](#63-automated-batch-queue--monitoring)
   - [Aggregating Benchmark Results](#64-aggregating-benchmark-results)
7. [Command-Line Arguments](#7-command-line-arguments)
8. [Citation](#8-citation)

---

## 1. Background & Motivation

**Kolmogorov-Arnold Networks (KANs)** have emerged as a promising alternative to traditional Multi-Layer Perceptrons (MLPs). Grounded in the Kolmogorov-Arnold representation theorem, KANs replace fixed nodal activation functions with learnable univariate activation functions on edges, parameterized via adjustable B-spline bases. This architectural shift delivers superior accuracy and symbolic interpretability for scientific machine learning and function approximation.

However, the distinct computational pattern of KANs—characterized by fine-grained B-spline basis evaluations and increased activation parameters—imposes new demands on computational systems. To understand how KAN scales on modern High-Performance Computing (HPC) platforms, this study presents a rigorous empirical scalability analysis of **PyTorch DistributedDataParallel (DDP)** on high-performance supercomputing nodes equipped with NVIDIA A100 GPUs and InfiniBand interconnects.

---

## 2. Experimental Benchmark Design

The benchmarking framework investigates four orthogonal scalability dimensions across 21 controlled test cases:

```
                    ┌──────────────────────────────────────────────┐
                    │      KAN Distributed Scalability Suite       │
                    └──────────────────────┬───────────────────────┘
                                           │
         ┌──────────────────┬──────────────┴─────┬──────────────────┐
         │                  │                    │                  │
         ▼                  ▼                    ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Strong Scaling  │ │  Weak Scaling   │ │  Communication  │ │  Model Scaling  │
│ 1 to 8 GPUs     │ │ 25K samples/GPU │ │ Intra-node vs.  │ │ Width: 32 ~ 256 │
│ (1 to 4 Nodes)  │ │ Linear Problem  │ │ Inter-node      │ │ Parameters:     │
│ Fixed 100K data │ │ Scaling (25-200K│ │ NCCL Overhead   │ │ ~2.7K to ~21.5K │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
```

1. **Strong Scaling**:
   - Fixed problem size ($N = 100,000$ samples, 40 epochs).
   - Evaluates speedup and parallel efficiency as hardware resources scale from 1 to 8 GPUs across 1 to 4 compute nodes.
2. **Weak Scaling**:
   - Fixed workload per GPU ($25,000$ samples/GPU), scaling total problem size linearly from 25K to 200K samples.
   - Evaluates per-GPU throughput retention and cluster throughput scalability.
3. **Communication Overhead & Topology Analysis**:
   - Compares intra-node (NVLink/PCIe) versus inter-node (InfiniBand HDR + NCCL) communication latency and overhead.
   - Separates forward-backward compute time from All-Reduce gradient synchronization time.
4. **Model Parameter Scaling**:
   - Fixed 4-GPU dual-node environment with varying hidden layer widths ($w \in \{32, 64, 128, 256\}$), scaling model parameters from ~2.7K to ~21.5K.
   - Evaluates the compute-to-communication ratio, parameter throughput, and peak memory usage.

---

## 3. Summary of Key Findings

Based on empirical runs conducted on the CESGA Supercomputing platform (NVIDIA A100 GPUs with InfiniBand HDR, archived in `results/`):

| Evaluation Dimension | Scale | Speedup / Throughput | Efficiency | Key Insights |
| :--- | :--- | :--- | :--- | :--- |
| **Strong Scaling** | 1 GPU $\to$ 8 GPUs (4 Nodes) | 1 GPU: 4,442 samples/s<br>8 GPUs: 26,617 samples/s | **5.97x Speedup**<br>**74.7%** Parallel Eff. | High scalability with minimal inter-node communication contention (overhead $\le$ 3.0% on 8 GPUs). |
| **Weak Scaling** | 1 GPU $\to$ 8 GPUs (25K/GPU) | Per-GPU throughput:<br>**3,440 ~ 3,462** samples/s | **99.3%** Throughput Retention | Near-ideal horizontal line, demonstrating KAN's readiness for large-scale data expansion. |
| **Communication** | Intra- vs. Inter-node (2~8 GPUs) | Intra-node: 289.24s<br>Inter-node (2N2G): 292.48s | **93.9% ~ 98.6%** Comm. Eff. | Intra-node overhead is negligible (~1.3%); InfiniBand keeps cross-node latency penalties remarkably low. |
| **Model Scaling** | Width: 32 $\to$ 256 (~2.7K~21.5K) | Per-GPU throughput: 6,155 $\to$ 867<br>VRAM: 0.04GB $\to$ 0.18GB | High Compute Density<br>Minimal VRAM footprint | Peak memory footprint remains under 0.2GB, showing that KAN training is compute-bound rather than memory-capacity-bound. |

---

## 4. Repository Structure

```text
.
├── README.md                      # Documentation (English & Chinese)
├── simple_main.py                 # Core driver: Single/DDP training, throughput, and communication profiling
├── simple_kan.py                  # KAN network implementation (pykan-based) and MLP baseline
├── simple_datasets.py             # Symbolic regression synthetic datasets (quadratic, polynomial, trig)
├── summarize_results.py           # Automated JSON parsing and summary table generation tool
├── auto_run_tests.sh              # Industrial automated SLURM batch job runner and queue monitor
├── simple_test.sh                 # Quick single-node verification script
├── test_report_20250720_212536.txt# Full execution audit log for all 21 benchmarking jobs
├── test_status.txt                # Job submission and completion status tracker
├── test_scripts/                  # 21 Standalone SLURM batch scripts
│   ├── strong_scaling_n*.sh       # Strong scaling scripts (1G1N through 8G4N)
│   ├── weak_scaling_n*.sh         # Weak scaling scripts (1G1N through 8G4N)
│   ├── communication_*.sh         # Intra-node and inter-node communication profiling scripts
│   └── model_scaling_*.sh         # Model width scaling scripts (Small, Medium, Large, XLarge)
├── results/                       # Complete raw experimental results and summary reports
│   ├── *_performance.json         # High-resolution hardware and training timing metrics
│   ├── *_training_results.csv     # Per-epoch loss and throughput telemetry logs
│   ├── strong_scaling_summary.txt # Formatted strong scaling metrics table
│   ├── weak_scaling_summary.txt   # Formatted weak scaling metrics table
│   ├── communication_summary.txt  # Formatted communication analysis table
│   └── model_scaling_summary.txt  # Formatted model scaling metrics table
└── logs/                          # SLURM output (.out) and error (.err) console logs
```

---

## 5. Environment Setup & Prerequisites

### Hardware & Software Stack
- **OS**: Linux (CentOS 7+, RHEL 8+, Ubuntu 20.04+)
- **Accelerator**: NVIDIA A100 (or V100/H100/RTX series, CUDA $\ge$ 11.8)
- **Interconnect**: NVLink/NVSwitch (intra-node), InfiniBand HDR/NDR (inter-node)
- **Workload Manager**: SLURM

### Conda Environment Installation
```bash
# 1. Create and activate a clean Conda environment
conda create -n pykan-env python=3.10 -y
conda activate pykan-env

# 2. Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install pykan and analysis dependencies
pip install pykan numpy pandas psutil pyyaml matplotlib
```

---

## 6. Quick Start & Reproduction Guide

### 6.1. Local & Single-Node Multi-GPU Execution

Run a quick test locally or on a single development node:

```bash
# Single-GPU test (default 50K samples, 40 epochs)
python simple_main.py --mode single --epochs 40 --batch-size 512 --width 64

# Single-node 2-GPU DDP test using torchrun
torchrun --nproc_per_node=2 simple_main.py \
    --mode ddp \
    --epochs 40 \
    --batch-size 512 \
    --test-type strong_scaling
```

### 6.2. Submitting Individual SLURM Batch Jobs

To submit specific benchmarks on a SLURM cluster:

```bash
# Submit 8-GPU 4-node strong scaling job
sbatch test_scripts/strong_scaling_n4g8.sh

# Submit 4-GPU dual-node inter-node communication test
sbatch test_scripts/communication_inter_n2g4.sh

# Submit extra-large model scaling test
sbatch test_scripts/model_scaling_xlarge_n2g4.sh
```

All test scripts automatically handle master address negotiation and apply recommended NCCL optimizations:
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0
```

### 6.3. Automated Batch Queue & Monitoring

The repository includes `auto_run_tests.sh`, which automates batch submission across all 21 test configurations, enforces concurrent job limits, checks execution status, and handles automatic retries:

```bash
chmod +x auto_run_tests.sh
./auto_run_tests.sh
```

Track real-time progress via:
```bash
cat test_status.txt
```

### 6.4. Aggregating Benchmark Results

After all tests finish, parse the telemetry JSON files and regenerate the performance tables:

```bash
python summarize_results.py
```

Summary tables will be generated directly in the `results/` directory (`*.txt` and `*.csv`).

---

## 7. Command-Line Arguments

Key parameters available in `simple_main.py`:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mode` | `str` | `single` | Training mode: `single` (single GPU) or `ddp` (Distributed Data Parallel) |
| `--test-type` | `str` | `strong_scaling` | Test scenario: `strong_scaling`, `weak_scaling`, `communication_overhead_analysis`, `model_scaling` |
| `--epochs` | `int` | `40` | Total training epochs (standardized across all benchmarks) |
| `--batch-size` | `int` | `512` | Per-GPU micro-batch size |
| `--width` | `int` | `64` | KAN hidden layer width |
| `--grid-size` | `int` | `10` | B-spline grid resolution |
| `--k` | `int` | `3` | B-spline degree ($k=3$ corresponds to cubic splines) |
| `--dataset-size` | `int` | `50000` | Dataset sample count |
| `--output-dir` | `str` | `./results` | Directory where metric JSONs and CSV logs are saved |

---

## 8. Citation

If you find this repository or benchmark results useful in your research, please cite our paper:

```bibtex
@article{kan_scalability_ddp_2026,
  title={Scalability Analysis of Distributed Kolmogorov-Arnold Network Training on High-Performance Computing Systems},
  author={Guangneng Chen and Contributors},
  journal={High-Performance Computing and Scientific Machine Learning},
  year={2026},
  note={Official Source Code and Benchmarking Artifacts}
}
```

---
---

# 🇨🇳 中文版本

> **论文官方源代码说明**：  
> 本代码仓库是学术论文 **《Scalability Analysis of Distributed Kolmogorov-Arnold Network Training on High-Performance Computing Systems》** 的官方完整源代码、自动化基准测试套件与实验结果归档。

---

## 📖 目录

1. [项目背景与动机](#1-项目背景与动机)
2. [实验评测体系设计](#2-实验评测体系设计)
3. [核心实验结论摘要](#3-核心实验结论摘要)
4. [项目目录结构](#4-项目目录结构)
5. [环境依赖与安装配置](#5-环境依赖与安装配置)
6. [运行与复现指南](#6-运行与复现指南)
   - [本地与单机单卡/多卡运行](#61-本地与单机单卡多卡运行)
   - [SLURM 集群批量脚本提交](#62-slurm-集群批量脚本提交)
   - [自动化批量调度与监控](#63-自动化批量调度与监控)
   - [实验结果汇总生成](#64-实验结果汇总生成)
7. [关键命令行参数说明](#7-关键命令行参数说明)
8. [论文引用](#8-论文引用)

---

## 1. 项目背景与动机

**柯尔莫哥洛夫-阿诺德网络（Kolmogorov-Arnold Networks, KAN）** 作为传统多层感知机（MLP）的一种强力替代架构，基于柯尔莫哥洛夫-阿诺德表示定理，将传统神经元节点上的固定激活函数替换为网络边上可学习的单变量激活函数，并通过可调节的 B-样条基函数进行参数化。该架构在符号回归、偏微分方程数值逼近以及科学机器学习（Scientific ML）中展现出了更高的函数拟合精度与可解释性。

然而，KAN 独特的计算模式（细粒度的 B-样条基函数评估以及参数结构）对底层计算硬件与并行架构提出了全新挑战。为了系统评估 KAN 在现代高性能计算（HPC）集群上的横向扩展能力，本研究在配备 NVIDIA A100 GPU 及 InfiniBand 高速互联的超算系统上，针对 **PyTorch DistributedDataParallel (DDP)** 分布式训练构建了完整的可扩展性评测基准体系。

---

## 2. 实验评测体系设计

测试套件围绕分布式计算的四大关键维度展开设计，共包含 21 组严格控制变量的分布式评测用例：

```
                    ┌──────────────────────────────────────────────┐
                    │      KAN 分布式可扩展性评测基准套件            │
                    └──────────────────────┬───────────────────────┘
                                           │
         ┌──────────────────┬──────────────┴─────┬──────────────────┐
         │                  │                    │                  │
         ▼                  ▼                    ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  强扩展性分析    │ │  弱扩展性分析    │ │  通信开销与拓扑  │ │  模型参数规模扩展│
│ (Strong Scaling)│ │ (Weak Scaling)  │ │ (Communication) │ │ (Model Scaling) │
│ 1G ~ 8G (4 节点)│ │ 每卡固定 25K 样本│ │ 节点内 vs 跨节点│ │ 隐藏宽度: 32~256│
│ 固定 100K 样本  │ │ 规模随卡数线性增│ │ NCCL 梯度同步耗时│ │ 参数: ~2.7K~21.5K│
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
```

1. **强扩展性分析 (Strong Scaling)**：
   - 固定总问题规模（100,000 样本，统一训练 40 轮次）。
   - 评测计算硬件从 1 GPU 扩展至 8 GPU（跨 1 到 4 个计算节点）下的加速比（Speedup）与并行效率（Parallel Efficiency）。
2. **弱扩展性分析 (Weak Scaling)**：
   - 固定每个 GPU 的计算负载（25,000 样本/GPU），总数据集随 GPU 规模成倍扩展（25K 至 200K 样本）。
   - 评估每张 GPU 吞吐量保持率与集群线性吞吐能力。
3. **通信开销与拓扑分析 (Communication Overhead & Topology Analysis)**：
   - 细粒度对比节点内（Intra-node, NVLink/PCIe）与跨节点（Inter-node, InfiniBand HDR + NCCL）的性能差异。
   - 分离前向/反向纯计算耗时与 All-Reduce 梯度同步通信耗时，量化通信瓶颈与重叠效率。
4. **模型容量扩展性 (Model Scaling)**：
   - 在固定的 4-GPU 双节点分布式环境下，系统性调节隐藏层宽度（Width: 32, 64, 128, 256），模型参数量从 ~2.7K 扩展到 ~21.5K。
   - 评估计算通信比、参数吞吐率及显存消耗特征。

---

## 3. 核心实验结论摘要

在 CESGA 超算集群（NVIDIA A100 40GB/64GB GPU，InfiniBand HDR）上的实际归档测算数据（详见 `results/` 目录）：

| 评测维度 | 测试规模 | 加速比 / 吞吐量表现 | 并行/通信效率 | 核心结论 |
| :--- | :--- | :--- | :--- | :--- |
| **强扩展性** | 1 GPU $\to$ 8 GPUs (4 节点) | 1 GPU: 4,442 samples/s<br>8 GPU: 26,617 samples/s | **加速比 5.97x**<br>并行效率 **74.7%** | 具有优异的横向强扩展性，8 卡跨 4 节点下的通信开销仅占 3.0%。 |
| **弱扩展性** | 1 GPU $\to$ 8 GPUs (25K/GPU) | 单卡吞吐量稳定在<br>**3,440 ~ 3,462** samples/s | 吞吐量保持率 **99.3%** | 弱扩展曲线近乎理想水平线，表明 KAN 架构高度适配大规模数据集的分布式训练。 |
| **通信分析** | 单节点 vs. 跨节点 (2~8 GPUs) | 节点内耗时: 289.24s<br>双节点耗时: 292.48s | 跨节点通信效率 **93.9% ~ 98.6%** | 节点内通信占比极低（~1.3%）；在 InfiniBand 和 NCCL 优化下跨节点附加开销极小。 |
| **模型扩展** | 宽度 32 $\to$ 256 (~2.7K~21.5K) | 单卡吞吐: 6,155 $\to$ 867<br>显存占用: 0.04GB $\to$ 0.18GB | 计算密度高<br>显存占用极小 | 显存峰值始终低于 0.2GB，表明 KAN 为典型的计算瓶颈架构，不存在大显存容量制约。 |

---

## 4. 项目目录结构

```text
.
├── README.md                      # 本说明文档（中英双语版）
├── simple_main.py                 # 核心驱动脚本：支持单机/DDP训练、吞吐量计算与通信开销分析
├── simple_kan.py                  # KAN 网络结构定义（基于 pykan 简化封装）与 MLP 对比基线
├── simple_datasets.py             # 符号回归基准数据集（二次曲面、多项式、三角函数）
├── summarize_results.py           # 自动化实验结果分析与对比报表生成脚本
├── auto_run_tests.sh              # 自动化批量作业提交、SLURM 任务队列监控与容错重试脚本
├── simple_test.sh                 # 快速单机/单节点验证测试脚本
├── test_report_20250720_212536.txt# 21 组基准测试全部通过的完整执行日志审计文件
├── test_status.txt                # 作业队列提交与完成状态跟踪记录
├── test_scripts/                  # 21 组面向 SLURM 的独立评测批处理脚本
│   ├── strong_scaling_n*.sh       # 强扩展性实验批处理脚本 (1G1N ~ 8G4N)
│   ├── weak_scaling_n*.sh         # 弱扩展性实验批处理脚本 (1G1N ~ 8G4N)
│   ├── communication_*.sh         # 节点内与跨节点通信开销测试脚本
│   └── model_scaling_*.sh         # 模型宽度扩展性测试脚本 (Small, Medium, Large, XLarge)
├── results/                       # 论文所有测试的完整原始数据与汇总报告 (71 个文件)
│   ├── *_performance.json         # 各测试用例的高精度耗时与硬件指标 JSON 文件
│   ├── *_training_results.csv     # 训练 Loss、吞吐量等轮次时序记录
│   ├── strong_scaling_summary.txt # 强扩展性最终对比汇总表
│   ├── weak_scaling_summary.txt   # 弱扩展性最终对比汇总表
│   ├── communication_summary.txt  # 通信开销最终对比汇总表
│   └── model_scaling_summary.txt  # 模型参数扩展最终对比汇总表
└── logs/                          # SLURM 标准输出 (.out) 与标准错误 (.err) 日志目录
```

---

## 5. 环境依赖与安装配置

### 硬件与软件基础环境
- **操作系统**: Linux (如 CentOS 7+, RHEL 8+, Ubuntu 20.04+ 等 HPC 发行版)
- **计算卡**: NVIDIA A100 (或 V100/H100/RTX 系列，CUDA $\ge$ 11.8)
- **网络互联**: NVLink / NVSwitch (节点内), InfiniBand HDR/NDR (跨节点)
- **集群调度器**: SLURM

### Conda 环境构建
```bash
# 1. 创建并激活 Python 3.10 环境
conda create -n pykan-env python=3.10 -y
conda activate pykan-env

# 2. 安装带 CUDA 加速支持的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装 pykan 与分析依赖
pip install pykan numpy pandas psutil pyyaml matplotlib
```

---

## 6. 运行与复现指南

### 6.1. 本地与单机单卡/多卡运行

在本地开发机或单节点上进行快速逻辑验证：

```bash
# 单 GPU 运行（默认 50K 样本，40 轮）
python simple_main.py --mode single --epochs 40 --batch-size 512 --width 64

# 使用 torchrun 进行单节点 2 卡 DDP 测试
torchrun --nproc_per_node=2 simple_main.py \
    --mode ddp \
    --epochs 40 \
    --batch-size 512 \
    --test-type strong_scaling
```

### 6.2. SLURM 集群批量脚本提交

进入 `test_scripts/` 目录，提交指定的评测作业：

```bash
# 提交 8-GPU 4-节点强扩展性测试作业
sbatch test_scripts/strong_scaling_n4g8.sh

# 提交 4-GPU 双节点通信开销测试作业
sbatch test_scripts/communication_inter_n2g4.sh

# 提交超大模型规模测试作业
sbatch test_scripts/model_scaling_xlarge_n2g4.sh
```

批处理脚本已内置主节点环境变量智能解析与 NCCL 最佳实践配置：
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0
```

### 6.3. 自动化批量调度与监控

使用本仓库提供的 `auto_run_tests.sh` 脚本可自动化按批次提交全部 21 组作业，具备队列控制与自动重试功能：

```bash
chmod +x auto_run_tests.sh
./auto_run_tests.sh
```

实时进度与作业状态查看：
```bash
cat test_status.txt
```

### 6.4. 实验结果汇总生成

当所有实验执行完毕后，调用汇总脚本，将 `results/` 中的测试原始数据提炼生成标准对比报告：

```bash
python summarize_results.py
```

结果报表将保存在 `results/` 目录下（包含 `.txt` 文本排版和 `.csv` 表格文件）。

---

## 7. 关键命令行参数说明

`simple_main.py` 核心参数解析：

| 参数名 | 类型 | 默认值 | 功能说明 |
| :--- | :--- | :--- | :--- |
| `--mode` | `str` | `single` | 运行模式：`single`（单 GPU）或 `ddp`（DistributedDataParallel） |
| `--test-type` | `str` | `strong_scaling` | 测试方案：`strong_scaling`、`weak_scaling`、`communication_overhead_analysis`、`model_scaling` |
| `--epochs` | `int` | `40` | 统一训练轮数（保证各规模对比的严格一致性） |
| `--batch-size` | `int` | `512` | 每个 GPU 的微批次（Batch Size）大小 |
| `--width` | `int` | `64` | KAN 隐藏层宽度 |
| `--grid-size` | `int` | `10` | B-样条基函数的网格划分数 |
| `--k` | `int` | `3` | B-样条阶数（$k=3$ 表示三次 B-样条） |
| `--dataset-size` | `int` | `50000` | 数据集样本总量 |
| `--output-dir` | `str` | `./results` | 结果与日志存储目录 |

---

## 8. 论文引用

如在您的学术研究、科研评测或工程实践中使用了本仓库代码或实验数据，请引用本论文：

```bibtex
@article{kan_scalability_ddp_2026,
  title={Scalability Analysis of Distributed Kolmogorov-Arnold Network Training on High-Performance Computing Systems},
  author={Guangneng Chen and Contributors},
  journal={High-Performance Computing and Scientific Machine Learning},
  year={2026},
  note={Official Source Code and Benchmarking Artifacts}
}
```

---

## 📄 开源许可与维护说明

- 本代码库遵循学术开源规范，仅供学术交流与科研使用。
- 如有任何关于分布式实验复现或 HPC 集群配置的问题，请查阅各脚本中的详细注释或联系作者。
