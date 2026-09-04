#!/bin/bash
#SBATCH -J weak_scaling_n1g1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gres=gpu:a100:1
#SBATCH -t 01:00:00
#SBATCH --mem=64GB
#SBATCH -o ./logs/weak_scaling_n1g1_%j.out
#SBATCH -e ./logs/weak_scaling_n1g1_%j.err

# 🔥 智能网络配置 - 修复分布式训练问题
if [[ $SLURM_NNODES -gt 1 ]]; then
    # 跨节点配置
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

echo "=== 弱扩展性测试（每GPU 25K样本）: weak_scaling_n1g1 ==="
echo "节点数: $SLURM_NNODES"
echo "总GPU数: 1"
echo "作业ID: $SLURM_JOB_ID"
echo "GPU类型: 普通A100节点"
echo "CPU核心: 32 (1×32)"
echo "描述: 1GPU(25K)"
echo "开始时间: $(date)"

# 弱扩展性测试（每GPU 25K样本）
python simple_main.py \
    --mode single \
    --epochs 40 \
    --batch-size 512 \
    --k 3 \
    --grid-size 10 \
    --width 64 \
    --test-type weak_scaling \
    --dataset-size 25000 \
    --benchmark-name "weak_scaling_n1g1" \
    --output-dir ./results

echo "=== 测试完成 ==="
echo "结束时间: $(date)"

# 显示结果
echo "查找结果文件..."
result_files=$(find ./results -name "*weak_scaling_n1g1*_performance.json" | head -1)
if [ -n "$result_files" ]; then
    echo "SUCCESS: 测试结果已保存到: $result_files"
    echo "RESULTS: 快速查看"
    grep -E "总训练时间|并行效率|通信开销" ./results/*weak_scaling_n1g1*_summary.txt 2>/dev/null || echo "等待生成summary文件..."
else
    echo "WARNING: 未找到结果文件"
fi
