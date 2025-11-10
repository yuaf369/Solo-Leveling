import torch.nn as nn
from PositionalEncoder import PositionalEncoder
from NormLayer import NormLayer
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward

class Encoder(nn.Module):
    """
    Transformer 编码器
    将输入序列映射为一系列富含上下文信息的连续表示
    """

    def __init__(self, vocab_size, d_model, N, heads, dropout):
        """
        初始化编码器

        Args:
            vocab_size (int): 词汇表大小
            d_model (int): 模型维度（词嵌入维度）
            N (int): 编码器层的堆叠层数
            heads (int): 多头注意力的头数
            dropout (float): Dropout比率
        """
        super().__init__()
        self.N = N  # 编码器层数
        
        # 词嵌入层：将离散的词汇索引转换为连续的向量表示
        # 输入: [batch_size, seq_len] 的整数张量
        # 输出: [batch_size, seq_len, d_model] 的浮点数张量
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # 位置编码器：为序列注入位置信息
        self.pe = PositionalEncoder(d_model)
        
        # 创建N个相同的编码器层
        # ModuleList用于正确注册子模块，使其参数可被优化器识别
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout) for _ in range(N)])
        
        # 最终的层归一化
        self.norm = NormLayer(d_model)

    def forward(self, src, mask):
        """
        前向传播

        Args:
            src (Tensor): 源语言输入序列，形状 [batch_size, src_seq_len]
            mask (Tensor): 注意力掩码，用于遮挡padding位置等

        Returns:
            Tensor: 编码后的表示，形状 [batch_size, src_seq_len, d_model]
        """
        # 1. 词嵌入：将词汇索引转换为向量
        # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        x = self.embed(src)
        
        # 2. 添加位置编码
        # 让模型知道每个词在序列中的位置
        x = self.pe(x)
        
        # 3. 通过N个编码器层进行深层特征提取
        # 每一层都会进一步整合上下文信息
        for layer in self.layers:
            x = layer(x, mask)
            
        # 4. 最终层归一化，稳定输出表示
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    单个编码器层
    包含两个子层：多头自注意力 + 前馈神经网络
    每个子层都使用残差连接和层归一化
    """

    def __init__(self, d_model, heads, dropout=0.1):
        """
        初始化编码器层

        Args:
            d_model (int): 模型维度
            heads (int): 注意力头数
            dropout (float): Dropout比率
        """
        super().__init__()
        # 第一个子层（自注意力）的层归一化
        self.norm_1 = NormLayer(d_model)
        # 第二个子层（前馈网络）的层归一化  
        self.norm_2 = NormLayer(d_model)
        
        # 多头自注意力机制
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        
        # 前馈神经网络
        self.ff = FeedForward(d_model, dropout=dropout)
        
        # Dropout层，用于两个子层的输出
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        编码器层的前向传播

        Args:
            x (Tensor): 输入张量，形状 [batch_size, seq_len, d_model]
            mask (Tensor): 自注意力掩码

        Returns:
            Tensor: 输出张量，形状与输入相同
        """
        # 子层1：多头自注意力（使用Pre-Norm结构）
        # Pre-Norm: 先归一化再进入子层，训练更稳定
        x2 = self.norm_1(x)  # 层归一化
        # 自注意力：Q、K、V都来自编码器自身的输出
        attn_output = self.attn(x2, x2, x2, mask)  # 自注意力
        # 残差连接 + Dropout: x = x + Dropout(Attention(LayerNorm(x)))
        x = x + self.dropout_1(attn_output)
        
        # 子层2：前馈神经网络（同样使用Pre-Norm结构）
        x2 = self.norm_2(x)  # 层归一化
        ff_output = self.ff(x2)  # 前馈网络
        # 残差连接 + Dropout: x = x + Dropout(FFN(LayerNorm(x)))
        x = x + self.dropout_2(ff_output)
        
        return x