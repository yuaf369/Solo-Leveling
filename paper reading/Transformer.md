## <font style="color:rgb(25, 27, 31);">前言</font>
论文链接：[https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

<font style="color:rgb(15, 17, 21);">Transformer 提出一种</font>**<font style="color:rgb(15, 17, 21);">完全基于注意力机制</font>**<font style="color:rgb(15, 17, 21);">的序列转换模型，摒弃了传统的 RNN 和 CNN 结构，实现了更强的并行能力与更优的翻译效果。</font>

## <font style="color:rgb(15, 17, 21);">研究背景</font>
### <font style="color:rgb(15, 17, 21);">问题定义与动机</font>
+ **<font style="color:rgb(15, 17, 21);">任务</font>**<font style="color:rgb(15, 17, 21);">：序列转换（如机器翻译），将输入序列映射为输出序列。</font>
+ **<font style="color:rgb(15, 17, 21);">传统方法</font>**<font style="color:rgb(15, 17, 21);">：基于 RNN/LSTM 的编码器-解码器结构，存在</font>**<font style="color:rgb(15, 17, 21);">顺序计算瓶颈</font>**<font style="color:rgb(15, 17, 21);">，难以并行。</font>
+ **<font style="color:rgb(15, 17, 21);">注意力机制</font>**<font style="color:rgb(15, 17, 21);">：虽能缓解长距离依赖，但仍与 RNN 耦合。</font>

### <font style="color:rgb(15, 17, 21);">意义与时机</font>
+ **<font style="color:rgb(15, 17, 21);">并行化需求</font>**<font style="color:rgb(15, 17, 21);">：硬件（如 GPU）发展推动模型并行化。</font>
+ **<font style="color:rgb(15, 17, 21);">长序列建模</font>**<font style="color:rgb(15, 17, 21);">：传统 RNN 难以捕捉长距离依赖。</font>
+ **<font style="color:rgb(15, 17, 21);">突破点</font>**<font style="color:rgb(15, 17, 21);">：完全基于注意力，实现全局依赖建模。</font>

```plain
RNN: [词1] → [词2] → ... → [词n]   （串行）
Transformer: [所有词] ↔ [所有词]   （并行，全连接注意力）
```

## 核心思想
+ **<font style="color:rgb(15, 17, 21);">Self-Attention</font>**<font style="color:rgb(15, 17, 21);">：每个词与序列中所有词交互，计算加权表示。</font>
+ **<font style="color:rgb(15, 17, 21);">Multi-Head Attention</font>**<font style="color:rgb(15, 17, 21);">：多个注意力头捕捉不同子空间的语义信息。</font>
+ **<font style="color:rgb(15, 17, 21);">Positional Encoding</font>**<font style="color:rgb(15, 17, 21);">：引入位置信息，弥补无卷积/递归的缺陷。</font>



![](https://cdn.nlark.com/yuque/0/2025/png/56086314/1762743321335-92282b8d-491b-4b40-8dda-3ed4630e9a99.png)



Transformers由Encoder和Decoder两个部分组成，Encoder和Decoder都包含6个block。Transformer的工作流程如下：

**第一步**：<font style="color:rgb(25, 27, 31);">获取输入句子的每一个单词的表示向量 </font>**<font style="color:rgb(25, 27, 31);">X</font>**<font style="color:rgb(25, 27, 31);">，</font>**<font style="color:rgb(25, 27, 31);">X</font>**<font style="color:rgb(25, 27, 31);">由单词的 Embedding（Embedding就是从原始数据提取出来的Feature） 和单词位置的 Embedding 相加得到。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/56086314/1762746767278-9ac694ab-bc27-472c-a594-732b623c15f7.png)

**<font style="color:rgb(25, 27, 31);">第二步：</font>**<font style="color:rgb(25, 27, 31);">将得到的单词表示向量矩阵 (如上图所示，每一行是一个单词的表示 </font>**<font style="color:rgb(25, 27, 31);">x</font>**<font style="color:rgb(25, 27, 31);">) 传入 Encoder 中，经过 6 个 Encoder block 后可以得到句子所有单词的编码信息矩阵 </font>**<font style="color:rgb(25, 27, 31);">C</font>**<font style="color:rgb(25, 27, 31);">，如下图。单词向量矩阵用 </font>![image](https://cdn.nlark.com/yuque/__latex/6f64ed9c566194f18f62358a04af11a6.svg)<font style="color:rgb(25, 27, 31);">表示， n 是句子中单词个数，d 是表示向量的维度 (论文中 d=512)。每一个 Encoder block 输出的矩阵维度与输入完全一致。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/56086314/1762746950448-0ecf75d9-a7a5-4c1e-a829-78510a358733.png)



### <font style="color:rgb(15, 17, 21);">关键公式与解释</font>
#### <font style="color:rgb(15, 17, 21);">公式 (1)：Scaled Dot-Product Attention</font>
![image](https://cdn.nlark.com/yuque/__latex/946070a2b93c65ac340c3b2b95789033.svg)

| <font style="color:rgb(15, 17, 21);">符号</font> | <font style="color:rgb(15, 17, 21);">含义</font> |
| --- | --- |
| <font style="color:rgb(15, 17, 21);"></font>_<font style="color:rgb(15, 17, 21);">Q</font>_ | <font style="color:rgb(15, 17, 21);">查询矩阵</font> |
| <font style="color:rgb(15, 17, 21);"></font>_<font style="color:rgb(15, 17, 21);">K</font>_ | <font style="color:rgb(15, 17, 21);">键矩阵</font> |
| <font style="color:rgb(15, 17, 21);"></font>_<font style="color:rgb(15, 17, 21);">V</font>_ | <font style="color:rgb(15, 17, 21);">值矩阵</font> |
| <font style="color:rgb(15, 17, 21);"></font>_<font style="color:rgb(15, 17, 21);">dk</font>_ | <font style="color:rgb(15, 17, 21);">键维度</font> |
| <font style="color:rgb(15, 17, 21);"></font>_<font style="color:rgb(15, 17, 21);">dk</font>_ | <font style="color:rgb(15, 17, 21);">缩放因子，防止梯度消失</font> |




#### <font style="color:rgb(15, 17, 21);">公式 (2)：Multi-Head Attention</font>
![image](https://cdn.nlark.com/yuque/__latex/34ce9c6df7c63fbd1ee1970719c5ff46.svg)

![image](https://cdn.nlark.com/yuque/__latex/60cf9a2661c3be3229e570ac2c0a56c1.svg)



#### <font style="color:rgb(15, 17, 21);">公式 (3)：位置编码（正弦/余弦）</font>
![image](https://cdn.nlark.com/yuque/__latex/dbe34879596bf31bf1a117c56514abf0.svg)

![image](https://cdn.nlark.com/yuque/__latex/e009867d1fdab4bdcdd049ba91de7a98.svg)



### 伪代码
```python
def transformer_forward(x, y):
    # 编码器
    for layer in encoder_layers:
        x = layer.multi_head_attention(x, x, x)  # self-attention
        x = layer.feed_forward(x)

    # 解码器
    for layer in decoder_layers:
        y = layer.masked_multi_head_attention(y, y, y)  # masked self-attention
        y = layer.multi_head_attention(y, x, x)  # encoder-decoder attention
        y = layer.feed_forward(y)

    return softmax(linear(y))
```

## <font style="color:rgb(15, 17, 21);">实验分析</font>
### <font style="color:rgb(15, 17, 21);">数据集与设置</font>
+ **<font style="color:rgb(15, 17, 21);">WMT 2014</font>**<font style="color:rgb(15, 17, 21);">：英德（4.5M 句对）、英法（36M 句对）</font>
+ **<font style="color:rgb(15, 17, 21);">词表示</font>**<font style="color:rgb(15, 17, 21);">：Byte-Pair Encoding（BPE）</font>
+ **<font style="color:rgb(15, 17, 21);">硬件</font>**<font style="color:rgb(15, 17, 21);">：8×P100 GPUs</font>
+ **<font style="color:rgb(15, 17, 21);">训练时间</font>**<font style="color:rgb(15, 17, 21);">：Base 模型 12 小时，Big 模型 3.5 天</font>

### <font style="color:rgb(15, 17, 21);">主要结果表（BLEU 分数）</font>
| <font style="color:rgb(15, 17, 21);">模型</font> | <font style="color:rgb(15, 17, 21);">EN-DE</font> | <font style="color:rgb(15, 17, 21);">EN-FR</font> | <font style="color:rgb(15, 17, 21);">训练成本（FLOPs）</font> |
| --- | --- | --- | --- |
| <font style="color:rgb(15, 17, 21);">ByteNet</font> | <font style="color:rgb(15, 17, 21);">23.75</font> | <font style="color:rgb(15, 17, 21);">—</font> | <font style="color:rgb(15, 17, 21);">—</font> |
| <font style="color:rgb(15, 17, 21);">ConvS2S</font> | <font style="color:rgb(15, 17, 21);">25.16</font> | <font style="color:rgb(15, 17, 21);">40.46</font> | <font style="color:rgb(15, 17, 21);">9.6e18</font> |
| <font style="color:rgb(15, 17, 21);">GNMT + RL</font> | <font style="color:rgb(15, 17, 21);">24.6</font> | <font style="color:rgb(15, 17, 21);">39.92</font> | <font style="color:rgb(15, 17, 21);">2.3e19</font> |
| **<font style="color:rgb(15, 17, 21);">Transformer (base)</font>** | **<font style="color:rgb(15, 17, 21);">27.3</font>** | **<font style="color:rgb(15, 17, 21);">38.1</font>** | **<font style="color:rgb(15, 17, 21);">3.3e18</font>** |
| **<font style="color:rgb(15, 17, 21);">Transformer (big)</font>** | **<font style="color:rgb(15, 17, 21);">28.4</font>** | **<font style="color:rgb(15, 17, 21);">41.0</font>** | <font style="color:rgb(15, 17, 21);">2.3e19</font> |


<font style="color:rgb(15, 17, 21);">Transformer 在翻译任务上显著优于所有基线模型。</font>

<font style="color:rgb(15, 17, 21);"></font>

## <font style="color:rgb(15, 17, 21);">总结</font>
### <font style="color:rgb(15, 17, 21);">贡献点</font>
+ <font style="color:rgb(15, 17, 21);">提出</font>**<font style="color:rgb(15, 17, 21);">完全基于注意力</font>**<font style="color:rgb(15, 17, 21);">的序列转换模型。</font>
+ <font style="color:rgb(15, 17, 21);">实现</font>**<font style="color:rgb(15, 17, 21);">高度并行化</font>**<font style="color:rgb(15, 17, 21);">，训练速度大幅提升。</font>
+ <font style="color:rgb(15, 17, 21);">在两大翻译任务上达到</font>**<font style="color:rgb(15, 17, 21);">SOTA</font>**<font style="color:rgb(15, 17, 21);">，且模型更轻量。</font>

<font style="color:rgb(15, 17, 21);"></font>

### <font style="color:rgb(15, 17, 21);">优势 vs 局限</font>
| <font style="color:rgb(15, 17, 21);">优势</font> | <font style="color:rgb(15, 17, 21);">局限</font> |
| --- | --- |
| <font style="color:rgb(15, 17, 21);">并行能力强</font> | <font style="color:rgb(15, 17, 21);">计算复杂度 O(n²) 对长序列不友好</font> |
| <font style="color:rgb(15, 17, 21);">长距离依赖建模优</font> | <font style="color:rgb(15, 17, 21);">位置编码依赖先验假设</font> |
| <font style="color:rgb(15, 17, 21);">模型可解释性较强</font> | <font style="color:rgb(15, 17, 21);">需要大量数据与算力</font> |




