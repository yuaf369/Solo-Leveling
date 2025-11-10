import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制 (Multi-Head Self-Attention)
    实现了Transformer中的核心注意力机制，将输入线性投影后分成多个头并行计算注意力。
    """

    def __init__(self, heads, d_model, dropout=0.1):
        """
        初始化多头注意力层

        Args:
            heads (int): 注意力头的数量
            d_model (int): 输入和输出的维度（也是词嵌入维度）
            dropout (float): Dropout比率，用于注意力权重的随机失活
        """
        super().__init__()
        
        # 确保d_model可以被heads整除，这样才能均匀分到每个头
        assert d_model % heads == 0, "d_model必须能被heads整除"
        
        self.d_model = d_model  # 模型维度
        self.d_k = d_model // heads  # 每个注意力头的维度
        self.h = heads  # 注意力头的数量
        
        # 定义Q、K、V的线性变换层，每个都将d_model维映射到d_model维
        self.q_linear = nn.Linear(d_model, d_model)  # 查询(Query)线性变换
        self.v_linear = nn.Linear(d_model, d_model)  # 值(Value)线性变换  
        self.k_linear = nn.Linear(d_model, d_model)  # 键(Key)线性变换
        
        self.dropout = nn.Dropout(dropout)  # 注意力dropout
        self.out = nn.Linear(d_model, d_model)  # 最终输出线性层

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        """
        计算缩放点积注意力

        Args:
            q: 查询张量, 形状 [batch_size, heads, seq_len, d_k]
            k: 键张量, 形状 [batch_size, heads, seq_len, d_k]  
            v: 值张量, 形状 [batch_size, heads, seq_len, d_k]
            d_k: 每个注意力头的维度
            mask: 注意力掩码，用于遮挡某些位置
            dropout: dropout层实例

        Returns:
            注意力输出张量, 形状 [batch_size, heads, seq_len, d_k]
        """
        # 计算Q和K的点积，然后缩放：scores = Q * K^T / sqrt(d_k)
        # 结果形状: [batch_size, heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 如果提供了掩码，应用掩码（将掩码位置设为极小的负数）
        if mask is not None:
            # 在heads维度上增加一个维度，以便广播
            mask = mask.unsqueeze(1)  # 形状: [batch_size, 1, seq_len, seq_len] 或类似
            # 将掩码为0的位置替换为-1e9，这样softmax后这些位置的权重接近0
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 在最后一个维度(seq_len)上应用softmax，得到注意力权重
        # 形状保持不变: [batch_size, heads, seq_len, seq_len]
        scores = F.softmax(scores, dim=-1)
        
        # 应用dropout到注意力权重上
        if dropout is not None:
            scores = dropout(scores)
        
        # 用注意力权重加权求和值向量 V
        # 矩阵乘法: [batch_size, heads, seq_len, seq_len] × [batch_size, heads, seq_len, d_k]
        # 输出形状: [batch_size, heads, seq_len, d_k]
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        """
        前向传播

        Args:
            q: 查询张量, 形状 [batch_size, q_seq_len, d_model]
            k: 键张量, 形状 [batch_size, k_seq_len, d_model]  
            v: 值张量, 形状 [batch_size, v_seq_len, d_model]
            mask: 注意力掩码，形状 [batch_size, seq_len] 或 [batch_size, seq_len, seq_len]

        Returns:
            多头注意力输出, 形状 [batch_size, q_seq_len, d_model]
        """
        bs = q.size(0)  # 批量大小 batch_size
        
        # 对K进行线性变换并重塑为多头格式
        # 1. 线性变换: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        # 2. 重塑: [batch_size, seq_len, d_model] -> [batch_size, seq_len, h, d_k]
        # 3. 转置: [batch_size, seq_len, h, d_k] -> [batch_size, h, seq_len, d_k]
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        
        # 对Q进行同样的操作
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        
        # 对V进行同样的操作
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        # 计算注意力，得到形状为 [batch_size, h, seq_len, d_k] 的输出
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        # 将多个头的输出拼接起来
        # 1. 转置: [batch_size, h, seq_len, d_k] -> [batch_size, seq_len, h, d_k]
        # 2. 保证内存连续: contiguous()
        # 3. 重塑: [batch_size, seq_len, h, d_k] -> [batch_size, seq_len, d_model] (因为 h * d_k = d_model)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        # 通过最终的线性变换层
        output = self.out(concat)
        
        return output