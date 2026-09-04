#!/bin/bash
#SBATCH -J communication_inter_n4g4
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gres=gpu:a100:1
#SBATCH -t 01:30:00
#SBATCH --mem=64GB
#SBATCH -o ./logs/communication_inter_n4g4_%j.out
#SBATCH -e ./logs/communication_inter_n4g4_%j.err

# 🔥 智能网络配置 - 修复分布式训练问题
if [[ $SLURM_NNODES -gt 1 ]]; then
    # 跨节点配置 (multi-node)
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_ADDR=$master_addr
    export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
    echo "跨节点DDP配置: Master=$MASTER_ADDR:$MASTER_PORT"
    
    # NCCL优化设置
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_P2P_DISABLE=0
else
    # 单节点配置
    export MASTER_ADDR=localhost
    export MASTER_PORT=12355
    echo "单节点DDP配置: Master=$MASTER_ADDR:$MASTER_PORT"
fi
export WORLD_SIZE=$SLURM_NPROCS
export OMP_NUM_THREADS=32

# 加载模块
module purge
module load cesga/system miniconda3/22.11.1-1

# 激活conda环境
conda activate $STORE/kan/envs/pykan-env

# 设置环境变量
export PYTHONNOUSERSITE=1

# 创建输出目录
mkdir -p results logs

echo "=== 通信开销测试（固定50K样本）: communication_inter_n4g4 ==="
echo "节点数: $SLURM_NNODES"
echo "总GPU数: 4"
echo "作业ID: $SLURM_JOB_ID"
echo "GPU类型: 普通A100节点"
echo "CPU核心: 128 (4×32)"
echo "描述: 4GPU四节点间通信"
echo "开始时间: $(date)"
echo "节点列表: $SLURM_JOB_NODELIST"
echo "Master节点: $MASTER_ADDR"
echo "Master端口: $MASTER_PORT"

echo "检查节点连通性..."
srun hostname | sort

# 通信开销测试（固定50K样本）
srun python simple_main.py \
    --mode ddp \
    --epochs 40 \
    --batch-size 512 \
    --k 3 \
    --grid-size 10 \
    --width 64 \
    --test-type communication_overhead_analysis \
    --dataset-size 50000 \
    --benchmark-name "communication_inter_n4g4" \
    --output-dir ./results

echo "测试完成时间: $(date)"

# 检查结果文件
echo "生成的结果文件:"
ls -la results/communication_inter_n4g4_* 2>/dev/null || echo "未找到结果文件"

# 显示GPU内存使用情况
echo "GPU内存使用情况:"
srun --ntasks=4 nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "无法获取GPU信息"

echo "作业完成: $(date)"
