import torch.nn as nn
from PositionalEncoder import PositionalEncoder
from NormLayer import NormLayer
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward

class Decoder(nn.Module):
    """
    Transformer 解码器
    基于编码器输出和已生成的目标序列，自回归地生成下一个词元
    """

    def __init__(self, vocab_size, d_model, N, heads, dropout):
        """
        初始化解码器

        Args:
            vocab_size (int): 目标语言词汇表大小
            d_model (int): 模型维度（词嵌入维度）
            N (int): 解码器层的堆叠层数
            heads (int): 多头注意力的头数
            dropout (float): Dropout比率
        """
        super().__init__()
        self.N = N  # 解码器层数
        
        # 目标语言的词嵌入层
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # 位置编码器：为目标序列注入位置信息
        self.pe = PositionalEncoder(d_model)
        
        # 创建N个相同的解码器层
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(N)])
        
        # 最终的层归一化
        self.norm = NormLayer(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        """
        前向传播

        Args:
            trg (Tensor): 目标语言输入序列（训练时是右移一个位置的目标序列）
                        形状 [batch_size, trg_seq_len]
            e_outputs (Tensor): 编码器的输出，形状 [batch_size, src_seq_len, d_model]
            src_mask (Tensor): 源序列掩码，用于遮挡padding位置
            trg_mask (Tensor): 目标序列掩码，用于防止看到未来信息（因果掩码）

        Returns:
            Tensor: 解码后的表示，形状 [batch_size, trg_seq_len, d_model]
        """
        # 1. 目标序列词嵌入
        # [batch_size, trg_seq_len] -> [batch_size, trg_seq_len, d_model]
        x = self.embed(trg)
        
        # 2. 添加位置编码
        x = self.pe(x)
        
        # 3. 通过N个解码器层进行深层特征提取
        # 每一层都会整合目标序列自身信息和编码器输出的源序列信息
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, trg_mask)
            
        # 4. 最终层归一化
        return self.norm(x)
    

class DecoderLayer(nn.Module):
    """
    单个解码器层
    包含三个子层：
    1. 掩码多头自注意力（目标序列自身）
    2. 编码器-解码器注意力（目标序列关注源序列）  
    3. 前馈神经网络
    每个子层都使用残差连接和层归一化
    """

    def __init__(self, d_model, heads, dropout=0.1):
        """
        初始化解码器层

        Args:
            d_model (int): 模型维度
            heads (int): 注意力头数
            dropout (float): Dropout比率
        """
        super().__init__()
        # 三个子层各自的层归一化
        self.norm_1 = NormLayer(d_model)  # 掩码自注意力前
        self.norm_2 = NormLayer(d_model)  # 编码器-解码器注意力前
        self.norm_3 = NormLayer(d_model)  # 前馈网络前
        
        # 三个Dropout层
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout) 
        self.dropout_3 = nn.Dropout(dropout)
        
        # 第一个注意力：掩码自注意力（目标序列内部）
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        
        # 第二个注意力：编码器-解码器注意力（目标查询源）
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        
        # 前馈神经网络
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        """
        解码器层的前向传播

        Args:
            x (Tensor): 目标序列的当前表示，形状 [batch_size, trg_seq_len, d_model]
            e_outputs (Tensor): 编码器输出，形状 [batch_size, src_seq_len, d_model]  
            src_mask (Tensor): 源序列掩码
            trg_mask (Tensor): 目标序列掩码（因果掩码）

        Returns:
            Tensor: 更新后的目标序列表示
        """
        # 子层1：掩码多头自注意力（目标序列自身）
        # 作用：让目标序列的每个位置关注之前的所有位置（不包括未来位置）
        x2 = self.norm_1(x)  # Pre-Norm
        # attn_1(Q, K, V, mask): 这里Q、K、V都来自目标序列，使用trg_mask
        masked_attn = self.attn_1(x2, x2, x2, trg_mask)
        # 残差连接
        x = x + self.dropout_1(masked_attn)
        
        # 子层2：编码器-解码器注意力（交叉注意力）
        # 作用：让目标序列的每个位置关注源序列的所有相关信息
        x2 = self.norm_2(x)  # Pre-Norm
        # attn_2(Q, K, V, mask): 
        # - Q: 来自目标序列（解码器上一层的输出）
        # - K, V: 来自编码器输出（源序列信息）
        # - mask: 使用src_mask遮挡源序列的padding位置
        cross_attn = self.attn_2(x2, e_outputs, e_outputs, src_mask)
        # 残差连接  
        x = x + self.dropout_2(cross_attn)
        
        # 子层3：前馈神经网络
        # 作用：对每个位置进行非线性变换，深化特征表示
        x2 = self.norm_3(x)  # Pre-Norm
        ff_output = self.ff(x2)
        # 残差连接
        x = x + self.dropout_3(ff_output)
        
        return x