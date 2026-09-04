# -*- coding: utf-8 -*-
"""
简化的KAN网络实现
基于pykan但极简化，去除复杂功能
"""

import torch
import torch.nn as nn
import numpy as np
from kan import KAN

class SimpleKAN(nn.Module):
    """简化的KAN网络"""
    
    def __init__(self, input_dim=2, hidden_width=64, k=3, grid_size=10):
        super().__init__()
        
        width = [input_dim, hidden_width, 1]
        
        self.kan = KAN(
            width=width,
            grid=grid_size,
            k=k,
            seed=42
        )
    
    def forward(self, x):
        return self.kan(x)

class SimpleMLPBaseline(nn.Module):
    """简单的MLP基线对比"""
    
    def __init__(self, input_dim=2, hidden_width=64):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 测试函数
if __name__ == "__main__":
    model = SimpleKAN(input_dim=2, hidden_width=64, k=3, grid_size=10)
    
    x = torch.randn(32, 2)
    y = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"参数数量: {count_parameters(model)}")
    
    mlp = SimpleMLPBaseline(input_dim=2, hidden_width=64)
    y_mlp = mlp(x)
    
    print(f"MLP输出形状: {y_mlp.shape}")
    print(f"MLP参数数量: {count_parameters(mlp)}") 