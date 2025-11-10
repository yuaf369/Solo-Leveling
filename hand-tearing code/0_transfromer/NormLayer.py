import torch
import torch.nn as nn

class NormLayer(nn.Module):
    """
    层归一化 (Layer Normalization)
    对每个样本的所有特征维度进行归一化，稳定训练过程并加速收敛。
    与批归一化(Batch Norm)不同，层归一化在序列模型中表现更好，对小批量大小不敏感。
    """

    def __init__(self, d_model, eps=1e-6):
        """
        初始化层归一化层

        Args:
            d_model (int): 输入特征的维度（词嵌入维度）
            eps (float): 一个小常数，添加到方差中以防止除以零
        """
        super().__init__()
        self.size = d_model  # 特征维度
        
        # 可学习的缩放参数（gamma），初始化为全1向量
        # 在归一化后重新缩放数据，让网络能够学习到合适的表示范围
        self.alpha = nn.Parameter(torch.ones(self.size))
        
        # 可学习的偏置参数（beta），初始化为全0向量  
        # 在归一化后重新调整数据的中心位置
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        # 数值稳定项，防止分母为零
        self.eps = eps

    def forward(self, x):
        """
        前向传播：应用层归一化

        Args:
            x (Tensor): 输入张量，形状为 [batch_size, seq_len, d_model] 
                        或 [batch_size, d_model] 等

        Returns:
            Tensor: 归一化后的张量，形状与输入相同
        """
        # 层归一化计算公式：
        # output = alpha * (x - mean) / (std + eps) + bias
        
        # 计算均值和标准差：
        # - dim=-1: 在最后一个维度（特征维度）上计算统计量
        # - keepdim=True: 保持维度，便于广播运算
        # 对于形状 [batch_size, seq_len, d_model] 的输入：
        #   mean形状: [batch_size, seq_len, 1]
        #   std形状:  [batch_size, seq_len, 1]
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm