import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    """
    Transformer 完整模型
    基于Encoder-Decoder架构的序列到序列(seq2seq)模型
    主要用于机器翻译等任务，完全基于自注意力机制
    """

    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        """
        初始化Transformer模型

        Args:
            src_vocab (int): 源语言词汇表大小
            trg_vocab (int): 目标语言词汇表大小
            d_model (int): 模型维度（词嵌入维度）
            N (int): Encoder和Decoder的堆叠层数
            heads (int): 多头注意力的头数
            dropout (float): Dropout比率
        """
        super().__init__()
        
        # 编码器：将源语言序列编码为上下文相关的表示
        # 输入: 源语言词索引 -> 输出: 富含上下文信息的向量表示
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        
        # 解码器：基于编码器输出和目标序列，自回归地生成翻译结果
        # 输入: 目标语言词索引 + 编码器输出 -> 输出: 目标语言的隐藏表示
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        
        # 输出线性层：将解码器的输出投影到目标语言词汇表空间
        # 输入: [batch_size, seq_len, d_model] -> 输出: [batch_size, seq_len, trg_vocab]
        # 后续会接Softmax得到每个位置的概率分布
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        """
        前向传播

        Args:
            src (Tensor): 源语言输入序列，形状 [batch_size, src_seq_len]
            trg (Tensor): 目标语言输入序列（训练时右移一位），形状 [batch_size, trg_seq_len]
            src_mask (Tensor): 源序列掩码，用于遮挡padding位置
            trg_mask (Tensor): 目标序列掩码，包含因果掩码防止信息泄露

        Returns:
            Tensor: 对目标词汇表的预测分数，形状 [batch_size, trg_seq_len, trg_vocab]
        """
        # 1. 编码阶段：源序列 -> 上下文表示
        # encoder输入: [batch_size, src_seq_len] 
        # encoder输出: [batch_size, src_seq_len, d_model]
        e_outputs = self.encoder(src, src_mask)
        
        # 2. 解码阶段：基于编码器输出和目标序列生成预测
        # decoder输入: 
        #   - trg: [batch_size, trg_seq_len] (目标序列，训练时右移一位)
        #   - e_outputs: [batch_size, src_seq_len, d_model] (编码器输出)
        #   - src_mask: 源序列掩码
        #   - trg_mask: 目标序列掩码（因果掩码）
        # decoder输出: [batch_size, trg_seq_len, d_model]
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        
        # 3. 输出投影：将隐藏表示转换为词汇表分数
        # 输入: [batch_size, trg_seq_len, d_model]
        # 输出: [batch_size, trg_seq_len, trg_vocab]
        # 注意：这里没有Softmax，因为训练时通常使用CrossEntropyLoss自带Softmax
        output = self.out(d_output)
        
        return output