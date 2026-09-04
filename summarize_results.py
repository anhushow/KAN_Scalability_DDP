#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAN并行训练结果汇总脚本
将四种测试方案的结果汇总成四个表格
"""

import json
import pandas as pd
from pathlib import Path
import sys

def load_test_results(results_dir="./results"):
    
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return []
    
   
    json_files = list(results_path.glob("*_performance.json"))
    
    if not json_files:
        print(f"❌ {results_dir} 中未找到结果文件")
        return []
    
    print(f"📁 找到 {len(json_files)} 个结果文件")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            
            filename = json_file.stem.replace('_single_performance', '').replace('_ddp_performance', '').replace('_performance', '')
            
           
            result = {
                'test_name': filename,
                'total_training_time': data.get('total_training_time', 0),
                'avg_throughput': data.get('avg_throughput', 0),
                'per_gpu_throughput': data.get('per_gpu_throughput', 0),
                'avg_communication_overhead': data.get('avg_communication_overhead', 0),
                'peak_gpu_memory': data.get('peak_gpu_memory', 0),
                'final_loss': data.get('final_loss', 0),
                'total_params': data.get('total_params', 0),
                'config': data.get('config', {})
            }
            
           
            config = result['config']
            result['epochs'] = config.get('epochs', 40)
            result['batch_size'] = config.get('batch_size', 512)
            result['dataset_size'] = config.get('dataset_size', 0)
            result['width'] = config.get('width', 64)
            result['gpus'] = data.get('gpu_count', 1)
            result['nodes'] = data.get('node_count', 1)
            
            results.append(result)
            
        except Exception as e:
            print(f"⚠️  无法加载文件 {json_file}: {e}")
    
    print(f"✅ 成功加载 {len(results)} 个测试结果")
    return results

def categorize_tests(results):
    
    categories = {
        'strong_scaling': [],
        'weak_scaling': [],
        'communication': [],
        'model_scaling': []
    }
    
    for result in results:
        test_name = result['test_name']
        
        if test_name.startswith('strong_scaling'):
            categories['strong_scaling'].append(result)
        elif test_name.startswith('weak_scaling'):
            categories['weak_scaling'].append(result)
        elif test_name.startswith('communication'):
            categories['communication'].append(result)
        elif test_name.startswith('model_scaling'):
            categories['model_scaling'].append(result)
    
    return categories

def create_strong_scaling_table(results):
    
    if not results:
        return None
    
   
    results.sort(key=lambda x: x['gpus'])
    
    
    baseline_time = None
    for r in results:
        if r['gpus'] == 1:
            baseline_time = r['total_training_time']
            break
    
    table_data = []
    for r in results:
        speedup = baseline_time / r['total_training_time'] if baseline_time and r['total_training_time'] > 0 else 1.0
        efficiency = (speedup / r['gpus']) * 100 if r['gpus'] > 0 else 0
        
        table_data.append({
            '测试名称': r['test_name'],
            'GPU数量': r['gpus'],
            '节点数': r['nodes'],
            '训练时间(秒)': round(r['total_training_time'], 2),
            '总吞吐量(samples/s)': round(r['avg_throughput'], 1),
            '每GPU吞吐量(samples/s)': round(r['per_gpu_throughput'], 1),
            '加速比': round(speedup, 2),
            '并行效率(%)': round(efficiency, 1),
            '通信开销(%)': round(r['avg_communication_overhead'], 1),
            '峰值内存(GB)': round(r['peak_gpu_memory'], 2),
            '最终损失': round(r['final_loss'], 6)
        })
    
    return pd.DataFrame(table_data)

def create_weak_scaling_table(results):
    """创建弱扩展性测试结果表格"""
    if not results:
        return None
    
    
    results.sort(key=lambda x: x['gpus'])
    
    table_data = []
    for r in results:
        samples_per_gpu = r['dataset_size'] / r['gpus'] if r['gpus'] > 0 else 0
        
        table_data.append({
            '测试名称': r['test_name'],
            'GPU数量': r['gpus'],
            '节点数': r['nodes'],
            '数据集大小': r['dataset_size'],
            '每GPU样本数': int(samples_per_gpu),
            '训练时间(秒)': round(r['total_training_time'], 2),
            '总吞吐量(samples/s)': round(r['avg_throughput'], 1),
            '每GPU吞吐量(samples/s)': round(r['per_gpu_throughput'], 1),
            '通信开销(%)': round(r['avg_communication_overhead'], 1),
            '峰值内存(GB)': round(r['peak_gpu_memory'], 2),
            '最终损失': round(r['final_loss'], 6)
        })
    
    return pd.DataFrame(table_data)

def create_communication_table(results):
    """创建通信开销测试结果表格"""
    if not results:
        return None
    
   
    results.sort(key=lambda x: x['gpus'])
    
    table_data = []
    for r in results:
        
        comm_type = "节点内" if r['nodes'] == 1 else "节点间"
        
        table_data.append({
            '测试名称': r['test_name'],
            'GPU数量': r['gpus'],
            '节点数': r['nodes'],
            '通信类型': comm_type,
            '训练时间(秒)': round(r['total_training_time'], 2),
            '总吞吐量(samples/s)': round(r['avg_throughput'], 1),
            '每GPU吞吐量(samples/s)': round(r['per_gpu_throughput'], 1),
            '通信开销(%)': round(r['avg_communication_overhead'], 1),
            '通信效率(%)': round(100 - r['avg_communication_overhead'], 1),
            '峰值内存(GB)': round(r['peak_gpu_memory'], 2),
            '最终损失': round(r['final_loss'], 6)
        })
    
    return pd.DataFrame(table_data)

def create_model_scaling_table(results):
    """创建模型扩展性测试结果表格"""
    if not results:
        return None
    
   
    results.sort(key=lambda x: x['width'])
    
    table_data = []
    for r in results:
        
        model_size = "未知"
        if "small" in r['test_name']:
            model_size = "小型(~1K参数)"
        elif "medium" in r['test_name']:
            model_size = "中型(~4K参数)"
        elif "large" in r['test_name']:
            model_size = "大型(~16K参数)"
        elif "xlarge" in r['test_name']:
            model_size = "超大型(~64K参数)"

        
        per_param_throughput = r['avg_throughput'] / r['total_params'] if r['total_params'] > 0 else 0
        memory_efficiency = r['total_params'] / r['peak_gpu_memory'] if r['peak_gpu_memory'] > 0 else 0

        table_data.append({
            '测试名称': r['test_name'],
            '模型规模': model_size,
            '隐藏层宽度': r['width'],
            '总参数数': r['total_params'],
            'GPU数量': r['gpus'],
            '训练时间(秒)': round(r['total_training_time'], 2),
            '总吞吐量(samples/s)': round(r['avg_throughput'], 1),
            '每GPU吞吐量(samples/s)': round(r['per_gpu_throughput'], 1),
            '每参数吞吐量(samples/s/param)': round(per_param_throughput, 3),
            '通信开销(%)': round(r['avg_communication_overhead'], 1),
            '峰值内存(GB)': round(r['peak_gpu_memory'], 2),
            '内存效率(params/GB)': round(memory_efficiency, 0),
            '最终损失': round(r['final_loss'], 6)
        })
    
    return pd.DataFrame(table_data)

def save_tables_to_files(tables):
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    for test_type, df in tables.items():
        if df is not None and not df.empty:
            
            csv_file = output_dir / f"{test_type}_summary.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
            
            txt_file = output_dir / f"{test_type}_summary.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"KAN并行训练 - {test_type.replace('_', ' ').title()} 测试结果汇总\n")
                f.write("=" * 80 + "\n\n")
                f.write(df.to_string(index=False))
                f.write(f"\n\n总计: {len(df)} 个测试\n")
            
            print(f"✅ {test_type} 表格已保存:")
            print(f"   📄 CSV: {csv_file}")
            print(f"   📄 TXT: {txt_file}")

def main():
    
    print("=" * 60)
    print("📊 KAN并行训练结果汇总工具")
    print("=" * 60)
    
    
    results = load_test_results()
    
    if not results:
        print("❌ 没有找到测试结果，请先运行测试")
        sys.exit(1)
    
    
    categories = categorize_tests(results)
    
    
    tables = {
        'strong_scaling': create_strong_scaling_table(categories['strong_scaling']),
        'weak_scaling': create_weak_scaling_table(categories['weak_scaling']),
        'communication': create_communication_table(categories['communication']),
        'model_scaling': create_model_scaling_table(categories['model_scaling'])
    }
    
    
    print(f"\n📈 测试结果统计:")
    for test_type, df in tables.items():
        count = len(df) if df is not None else 0
        print(f"  {test_type.replace('_', ' ').title()}: {count} 个测试")
    
    
    print(f"\n💾 保存结果表格...")
    save_tables_to_files(tables)
    
    print(f"\n🎉 结果汇总完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
