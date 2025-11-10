import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    """
    Transformer 位置编码器 (Positional Encoder)。
    由于Transformer的自注意力机制本身不包含位置信息，此类通过注入正弦和余弦信号
    来为输入序列的每个位置提供顺序信息。
    """

    def __init__(self, d_model, max_seq_len=80):
        """
        初始化位置编码器。

        Args:
            d_model (int): 词嵌入向量的维度，也是位置编码的维度。
            max_seq_len (int): 预设支持的最大序列长度。
        """
        super().__init__()
        self.d_model = d_model

        # 初始化一个全零矩阵，用于存储所有位置的位置编码
        # 形状: [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)

        # 'pos' 是序列中的位置 (0, 1, 2, ... max_seq_len-1)
        # 创建一个形状为 [max_seq_len, 1] 的位置索引张量，用于后续计算
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 计算分母项：10000^(2i/d_model)
        # 这里使用了对数空间的技巧来高效稳定地计算指数项。
        # 首先计算 (2i / d_model)，注意i被整除2，所以维度减半。
        # 形状: [d_model/2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))

        # 应用正弦函数到偶数索引位置
        pe[:, 0::2] = torch.sin(pos * div_term)
        # 应用余弦函数到奇数索引位置
        pe[:, 1::2] = torch.cos(pos * div_term)

        # 为位置编码矩阵增加一个批次维度，以便与输入张量广播
        # 变形后形状: [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)

        # 将位置编码矩阵注册为模块的缓冲区(Buffer)
        # 缓冲区是模型的一部分，其参数会被保存，但不会被优化器更新（不是可学习参数）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播，将位置信息添加到输入张量中。

        Args:
            x (Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]

        Returns:
            Tensor: 添加了位置编码后的张量，形状与x相同 [batch_size, seq_len, d_model]
        """
        # 在添加位置编码之前，先将输入x放大（乘以sqrt(d_model)）
        # 这是因为后续会与位置编码相加，而位置编码的范数较小。
        # 这个操作有助于在相加后，保持词嵌入的原始语义信息占据主导地位。
        x = x * math.sqrt(self.d_model)

        # 获取输入序列的实际长度
        seq_len = x.size(1)

        # 将位置编码加到输入张量上。
        # 使用 `self.pe[:, :seq_len]` 是为了处理可变长度序列，只取所需长度的位置编码。
        # PyTorch的广播机制会自动将 [1, seq_len, d_model] 的pe加到 [batch_size, seq_len, d_model] 的x上。
        x = x + self.pe[:, :seq_len]

        return x