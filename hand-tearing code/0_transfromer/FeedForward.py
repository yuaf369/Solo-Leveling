import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    前馈神经网络 (Feed-Forward Network)
    也称为位置级前馈网络 (Position-wise Feed-Forward Network)
    这是Transformer中每个编码器和解码器层的重要组成部分。
    对序列中的每个位置独立且相同地应用两层线性变换和非线性激活。
    """

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        """
        初始化前馈网络层

        Args:
            d_model (int): 输入和输出的维度（词嵌入维度）
            d_ff (int): 中间隐藏层的维度，默认2048（在原始论文中为4*d_model）
            dropout (float): Dropout比率，用于防止过拟合
        """
        super().__init__()
        
        # 第一层线性变换：将输入从d_model维度扩展到更大的d_ff维度
        # 这种维度扩展为模型提供了更强的表示能力
        # 形状变化: [*, d_model] -> [*, d_ff]
        self.linear_1 = nn.Linear(d_model, d_ff)
        
        # Dropout层：在训练时随机"关闭"一部分神经元，增强模型泛化能力
        self.dropout = nn.Dropout(dropout)
        
        # 第二层线性变换：将维度从d_ff投影回d_model
        # 确保输出维度与输入维度一致，以便残差连接
        # 形状变化: [*, d_ff] -> [*, d_model]
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        前向传播

        Args:
            x (Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]

        Returns:
            Tensor: 输出张量，形状与输入相同 [batch_size, seq_len, d_model]
        """
        # 第一层变换 + ReLU激活 + Dropout
        # 计算流程: linear_1 → ReLU → Dropout
        # ReLU激活函数引入非线性，使模型能够学习更复杂的模式
        # Dropout在训练时随机将部分激活值设为0，防止过拟合
        x = self.dropout(F.relu(self.linear_1(x)))
        
        # 第二层线性变换（无激活函数）
        # 将维度投影回原始大小，准备与残差连接相加
        x = self.linear_2(x)
        
        return x