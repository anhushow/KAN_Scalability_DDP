#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAN并行化项目 - 最小可行性测试版本
支持单GPU、DistributedDataParallel两种模式的公平对比
包含通信开销测量和最小化测试配置
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import time
import os
import psutil
from pathlib import Path
import pandas as pd
import json
import math
from simple_kan import SimpleKAN
from simple_datasets import SimpleDataset



def parse_args():
    parser = argparse.ArgumentParser(description="KAN并行化最小可行性测试")
    
    parser.add_argument('--mode', type=str, choices=['single', 'ddp'], 
                       default='single', help="并行模式: single(单GPU), ddp(DistributedDataParallel)")
    
    parser.add_argument('--epochs', type=int, default=40, help="训练轮数(统一为40轮)")
    parser.add_argument('--batch-size', type=int, default=512, help="批次大小")
    parser.add_argument('--k', type=int, default=3, help="KAN网络K值")
    parser.add_argument('--grid-size', type=int, default=10, help="网格大小")
    parser.add_argument('--width', type=int, default=64, help="隐藏层宽度")
    parser.add_argument('--output-dir', type=str, default='./results', help="输出目录")
    
    parser.add_argument('--test-type', type=str, 
                       choices=['strong_scaling', 'weak_scaling', 'model_scaling', 'communication_overhead_analysis'],
                       default='strong_scaling', help="测试类型")
    
    parser.add_argument('--dataset-size', type=int, default=50000, 
                       help="数据集大小，默认50K用于强扩展性测试")
    
    parser.add_argument('--profile', action='store_true', help="启用性能分析")
    parser.add_argument('--benchmark-name', type=str, default='mvp_test', help="基准测试名称")
    
    return parser.parse_args()

def setup_distributed():

    if 'SLURM_PROCID' in os.environ:
        # SLURM环境
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        

        if 'MASTER_ADDR' not in os.environ:
        
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            print(f"Rank {rank}: 单节点DDP配置 - {world_size} GPUs")
        else:

            print(f"Rank {rank}: 跨节点DDP配置")
            print(f"Rank {rank}: Master节点: {os.environ['MASTER_ADDR']}")
            print(f"Rank {rank}: Master端口: {os.environ['MASTER_PORT']}")
        
        print(f"Rank {rank}: World size: {world_size}, Local rank: {local_rank}")
        print(f"Rank {rank}: 连接到 {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1, 0


def measure_communication_overhead(model, device, world_size, batch_size=512):
    """测量通信开销"""
    if world_size <= 1:
        return 0.0
    

    test_input = torch.randn(batch_size, 2).to(device)
    

    torch.cuda.synchronize()
    compute_start = time.time()
    
    with torch.no_grad():
        _ = model(test_input)
    
    torch.cuda.synchronize()
    compute_time = time.time() - compute_start
    
    torch.cuda.synchronize()
    total_start = time.time()
    
    with torch.no_grad():
        output = model(test_input)

        if hasattr(model, 'module'):
            for param in model.module.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
    
    torch.cuda.synchronize()
    total_time = time.time() - total_start
    

    communication_time = total_time - compute_time
    communication_overhead = (communication_time / total_time) * 100 if total_time > 0 else 0
    
    return max(0, communication_overhead)

def get_gpu_memory():

    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}

def get_system_metrics():
 
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_used_gb': psutil.virtual_memory().used / 1024**3
    }

def get_model_config(test_type, width=64):
 
    if test_type == 'model_scaling':
       
        configs = {
            'small': {'layers': [2, 32, 1], 'width': 32},        # ~1K参数
            'medium': {'layers': [2, 64, 64, 1], 'width': 64},   # ~4K参数
            'large': {'layers': [2, 128, 128, 1], 'width': 128}, # ~16K参数
            'xlarge': {'layers': [2, 256, 256, 1], 'width': 256} # ~64K参数
        }
        return configs
    else:
     
        return {'layers': [2, 64, 64, 1], 'width': width}

def train_epoch(model, dataloader, optimizer, criterion, device, world_size, profile=False):
  
    model.train()
    total_loss = 0.0
    batch_times = []
    communication_overheads = []
    
    epoch_start = time.time()
    
    for batch_idx, (X, y) in enumerate(dataloader):
        batch_start = time.time()
        
        X, y = X.to(device), y.to(device)
        
        
        compute_start = time.time()
        
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.squeeze(), y)
        loss.backward()
        
       
        if world_size > 1 and batch_idx % 2 == 0:
            comm_overhead = measure_communication_overhead(model, device, world_size, X.size(0))
            communication_overheads.append(comm_overhead)
        
        optimizer.step()
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        total_loss += loss.item()
        
       
        if profile and batch_idx % 100 == 0:
            gpu_mem = get_gpu_memory()
            print(f"  Batch {batch_idx}: Loss={loss.item():.6f}, "
                  f"Time={batch_time:.3f}s, GPU={gpu_mem['allocated']:.2f}GB")
    
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / len(dataloader)
    avg_comm_overhead = sum(communication_overheads) / len(communication_overheads) if communication_overheads else 0
    
    return avg_loss, epoch_time, batch_times, avg_comm_overhead

def main():
    args = parse_args()

   
    dataset_size = args.dataset_size

 
    performance_data = {
        'mode': args.mode,
        'test_type': args.test_type,
        'config': {
            'k': args.k,
            'grid_size': args.grid_size,
            'width': args.width,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'dataset_size': dataset_size
        },
        'gpu_count': 0,
        'node_count': 1,
        'metrics': []
    }

   
    rank, world_size, local_rank = setup_distributed()
    
   
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
   
    if args.mode == 'single':
        gpu_count = 1
        node_count = 1
    elif args.mode == 'ddp':
        gpu_count = world_size
        node_count = int(os.environ.get('SLURM_NNODES', 1))
    
    performance_data['gpu_count'] = gpu_count
    performance_data['node_count'] = node_count
    
    if rank == 0:
        print(f"=== KAN并行化最小可行性测试 ===")
        print(f"测试类型: {args.test_type}")
        print(f"模式: {args.mode.upper()}")
        print(f"数据集大小: {dataset_size:,}样本")
        print(f"参数: K={args.k}, Grid={args.grid_size}, Width={args.width}")
        print(f"设备: {device}, GPU数量: {gpu_count}, 节点数量: {node_count}")
        print(f"批次大小: {args.batch_size}, 训练轮数: {args.epochs}")
    
    
    dataset = SimpleDataset(dataset_size)
    
   
    sampler = None
    if args.mode == 'ddp' and world_size > 1:
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=(sampler is None), 
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    
    import shutil, datetime, uuid

    model_root = Path("./model_runs")  
    model_root.mkdir(exist_ok=True)


    job_id = os.getenv("SLURM_JOB_ID", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    unique_subdir = f"{args.benchmark_name}_{job_id}_{uuid.uuid4().hex[:8]}"
    current_model_dir = model_root / unique_subdir
    current_model_dir.mkdir(parents=True, exist_ok=True)

  
    try:
        if os.path.islink("./model") or os.path.exists("./model"):
           
            try:
                if os.path.islink("./model"):
                    os.unlink("./model")
                else:
                   
                    shutil.rmtree("./model", ignore_errors=True)
            except Exception as e:
                backup_name = f"./model_backup_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                print(f"[WARN] 无法删除旧 ./model，改为重命名备份: {backup_name} ({e})")
                try:
                    os.rename("./model", backup_name)
                except Exception as e2:
                    print(f"[WARN] 备份 ./model 失败: {e2}")
       
        os.symlink(current_model_dir.resolve(), "./model", target_is_directory=True)
    except Exception as e:
        print(f"[WARN] 创建 ./model 符号链接失败，将直接使用 {current_model_dir}: {e}")
        os.environ["KAN_CHECKPOINT_DIR"] = str(current_model_dir)
    
   
    model = SimpleKAN(input_dim=2, hidden_width=args.width, k=args.k, grid_size=args.grid_size)
    
    
    if args.mode == 'ddp' and world_size > 1:
        model = model.to(device)
        print(f"使用DistributedDataParallel，进程数: {world_size}")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:
        model = model.to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
   
    if hasattr(model, 'module'):
        total_params = sum(p.numel() for p in model.module.parameters())
    else:
        total_params = sum(p.numel() for p in model.parameters())
    
    if rank == 0:
        print(f"总参数数: {total_params:,}")
    
  
    results = []
    overall_start_time = time.time()
    
  
    if rank == 0:
        print("GPU预热中...")
    
   
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    for _ in range(3):
        X_dummy = torch.randn(args.batch_size, 2).to(device)
        _ = model(X_dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_start_time = time.time()
    
    total_communication_overhead = 0
    
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
    
        train_loss, epoch_time, batch_times, comm_overhead = train_epoch(
            model, dataloader, optimizer, criterion, device, world_size,
            profile=(args.profile and epoch % 5 == 0)  # 更频繁的监控
        )
        
        total_communication_overhead += comm_overhead
        
        
        if rank == 0:
            gpu_mem = get_gpu_memory()
            sys_metrics = get_system_metrics()
            
            
            processed_samples = math.ceil(len(dataloader.dataset) / world_size) if (args.mode == 'ddp' and world_size > 1) else len(dataloader.dataset)
            samples_per_second = processed_samples / epoch_time if epoch_time > 0 else 0

            epoch_data = {
                'epoch': epoch,
                'loss': train_loss,
                'epoch_time': epoch_time,
                'avg_batch_time': sum(batch_times) / len(batch_times),
                'samples_per_second': samples_per_second,
                'communication_overhead_percent': comm_overhead,
                'gpu_memory_gb': gpu_mem,
                'system_metrics': sys_metrics
            }
            
            results.append(epoch_data)
            performance_data['metrics'].append(epoch_data)
            
           
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                print(f"Epoch {epoch:2d}/{args.epochs}: "
                      f"Loss={train_loss:.6f}, "
                      f"Time={epoch_time:.2f}s, "
                      f"Throughput={epoch_data['samples_per_second']:.1f} samples/s, "
                      f"CommOverhead={comm_overhead:.1f}%, "
                      f"GPU={gpu_mem['allocated']:.2f}GB")
    
   
    if rank == 0:
        training_time = time.time() - training_start_time
        total_time = time.time() - overall_start_time
        
        per_gpu_throughput = sum(r['samples_per_second'] for r in results) / len(results) if results else 0
        avg_throughput = per_gpu_throughput * gpu_count
        
        avg_communication_overhead = total_communication_overhead / args.epochs if args.epochs > 0 else 0
        final_loss = results[-1]['loss'] if results else float('inf')
        peak_gpu_memory = max(r['gpu_memory_gb']['max_allocated'] for r in results) if results else 0
        
        print(f"\n=== 训练完成 ===")
        print(f"测试类型: {args.test_type}")
        print(f"总训练时间: {training_time:.2f}秒")
        print(f"平均吞吐量: {avg_throughput:.1f} samples/s (总计), {per_gpu_throughput:.1f} samples/s (每GPU)")
        print(f"平均通信开销: {avg_communication_overhead:.1f}%")
        print(f"最终损失: {final_loss:.6f}")
        print(f"峰值GPU内存: {peak_gpu_memory:.2f}GB")
        
       
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        
        df = pd.DataFrame(results)
        df.to_csv(output_dir / f'{args.benchmark_name}_{args.mode}_training_results.csv', index=False, lineterminator='\n')
        
       
        performance_data.update({
            'total_training_time': training_time,
            'total_time': total_time,
            'avg_throughput': avg_throughput,
            'per_gpu_throughput': per_gpu_throughput,
            'avg_communication_overhead': avg_communication_overhead,
            'final_loss': final_loss,
            'peak_gpu_memory': peak_gpu_memory,
            'total_params': total_params
        })
        
        with open(output_dir / f'{args.benchmark_name}_{args.mode}_performance.json', 'w', encoding='utf-8', newline='\n') as f:
            json.dump(performance_data, f, indent=2, ensure_ascii=False)
        
        
        with open(output_dir / f'{args.benchmark_name}_{args.mode}_summary.txt', 'w', encoding='utf-8', newline='\n') as f:
            f.write(f"KAN并行化最小可行性测试总结\n")
            f.write(f"{'='*50}\n")
            f.write(f"测试配置:\n")
            f.write(f"  测试类型: {args.test_type}\n")
            f.write(f"  并行模式: {args.mode.upper()}\n")
            f.write(f"  GPU数量: {gpu_count}\n")
            f.write(f"  节点数量: {node_count}\n")
            f.write(f"  数据集大小: {dataset_size:,}\n")
            f.write(f"  K值: {args.k}\n")
            f.write(f"  网格大小: {args.grid_size}\n")
            f.write(f"  隐藏层宽度: {args.width}\n")
            f.write(f"  批次大小: {args.batch_size}\n")
            f.write(f"  训练轮数: {args.epochs}\n")
            f.write(f"\n性能结果:\n")
            f.write(f"  总训练时间: {training_time:.2f}秒\n")
            f.write(f"  平均吞DDP模式吐量: {avg_throughput:.1f} samples/s (总计), {per_gpu_throughput:.1f} samples/s (每GPU)\n")
            f.write(f"  平均通信开销: {avg_communication_overhead:.1f}%\n")
            f.write(f"  最终损失: {final_loss:.6f}\n")
            f.write(f"  峰值GPU内存: {peak_gpu_memory:.2f}GB\n")
            f.write(f"  总参数数: {total_params:,}\n")
    
   
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 