#!/bin/bash
#SBATCH -J debug_min             
#SBATCH -o debug_%j.o            
#SBATCH -e debug_%j.e            
#SBATCH --gres=gpu:a100:1        
#SBATCH -c 32                    
#SBATCH -t 00:05:00              
#SBATCH --mem=8GB                

echo "=== 调试开始 ==="
echo "时间: $(date)"
echo "用户: $(whoami)"
echo "工作目录: $(pwd)"
echo "节点: $(hostname)"

# 设置CPU线程数
export OMP_NUM_THREADS=32

echo "=== 环境变量检查 ==="
echo "STORE: '$STORE'"
echo "HOME: '$HOME'"
echo "PATH: $PATH"

echo "=== 基础命令测试 ==="
echo "Bash版本: $BASH_VERSION"
which bash
which ls

echo "=== 目录创建测试 ==="
mkdir -p results logs test_output
ls -la
echo "目录创建: OK"

echo "=== 模块系统测试 ==="
echo "加载cesga/system..."
module purge
module load cesga/system
echo "cesga/system: OK"

echo "加载miniconda3..."
module load miniconda3/22.11.1-1
echo "miniconda3: OK"

echo "=== Conda测试 ==="
which conda
conda --version
echo "conda路径: $(which conda)"

echo "=== 环境列表 ==="
conda env list

echo "=== STORE路径检查 ==="
if [ -z "$STORE" ]; then
    echo "错误: STORE变量未设置"
    exit 1
else
    echo "STORE已设置: $STORE"
    ls -la $STORE/ 2>/dev/null || echo "警告: 无法访问STORE目录"
fi

echo "=== Conda环境路径检查 ==="
if [ -d "$STORE/kan/envs/pykan-env" ]; then
    echo "conda环境路径存在: $STORE/kan/envs/pykan-env"
else
    echo "错误: conda环境路径不存在: $STORE/kan/envs/pykan-env"
    echo "尝试查找可能的环境位置:"
    find $STORE -name "*pykan*" -type d 2>/dev/null || echo "未找到pykan环境"
fi

echo "=== 基础Python测试 ==="
which python
python --version

echo "=== 调试完成 ==="
echo "如果看到这条消息，说明脚本执行成功" 