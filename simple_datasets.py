# -*- coding: utf-8 -*-
"""
简化的数据集实现
只包含基本的符号回归函数，去除复杂的数据集
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import math

class SimpleDataset(Dataset):
    """基础符号回归数据集: f(x, y) = x² + y²"""
    
    def __init__(self, n_samples=10000, noise_level=0.01):
        self.n_samples = n_samples
        
        self.X = torch.rand(n_samples, 2) * 4 - 2
        
        self.y = (self.X[:, 0]**2 + self.X[:, 1]**2).float()
        
        if noise_level > 0:
            noise = torch.randn(n_samples) * noise_level
            self.y += noise
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PolynomialDataset(Dataset):
    """多项式数据集: f(x, y) = x³ + 2xy + y²"""
    
    def __init__(self, n_samples=10000, noise_level=0.01):
        self.n_samples = n_samples
        
        self.X = torch.rand(n_samples, 2) * 4 - 2
        
        x, y = self.X[:, 0], self.X[:, 1]
        self.y = (x**3 + 2*x*y + y**2).float()
        
        if noise_level > 0:
            noise = torch.randn(n_samples) * noise_level
            self.y += noise
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TrigDataset(Dataset):
    """三角函数数据集: f(x, y) = sin(x) + cos(y)"""
    
    def __init__(self, n_samples=10000, noise_level=0.01):
        self.n_samples = n_samples
        
        self.X = torch.rand(n_samples, 2) * 2 * math.pi
        
        x, y = self.X[:, 0], self.X[:, 1]
        self.y = (torch.sin(x) + torch.cos(y)).float()
        
        if noise_level > 0:
            noise = torch.randn(n_samples) * noise_level
            self.y += noise
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 数据集工厂函数
def get_dataset(name, n_samples=10000, **kwargs):
    datasets = {
        'simple': SimpleDataset,
        'polynomial': PolynomialDataset,
        'trigonometric': TrigDataset
    }
    
    if name.lower() in datasets:
        return datasets[name.lower()](n_samples, **kwargs)
    else:
        raise ValueError(f"未知数据集: {name}. 可用数据集: {list(datasets.keys())}")

# 测试函数
if __name__ == "__main__":
    print("测试数据集...")
    
    # 测试简单数据集
    dataset = SimpleDataset(1000)
    print(f"简单数据集: {len(dataset)} 样本")
    
    x, y = dataset[0]
    print(f"样本形状: X={x.shape}, y={y.shape}")
    print(f"样本值: X={x}, y={y}")
    
    # 测试多项式数据集
    poly_dataset = PolynomialDataset(1000)
    print(f"多项式数据集: {len(poly_dataset)} 样本")
    
    # 测试三角函数数据集
    trig_dataset = TrigDataset(1000)
    print(f"三角函数数据集: {len(trig_dataset)} 样本") 