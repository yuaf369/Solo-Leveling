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

**<font style="color:rgb(25, 27, 31);">第三步</font>**<font style="color:rgb(25, 27, 31);">：将 Encoder 输出的编码信息矩阵 </font>**<font style="color:rgb(25, 27, 31);">C</font>**<font style="color:rgb(25, 27, 31);">传递到 Decoder 中，Decoder 依次会根据当前翻译过的单词 1~ i 翻译下一个单词 i+1，如下图所示。在使用的过程中，翻译到单词 i+1 的时候需要通过 </font>**<font style="color:rgb(25, 27, 31);">Mask (掩盖)</font>**<font style="color:rgb(25, 27, 31);"> 操作遮盖住 i+1 之后的单词。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/56086314/1762754106336-07d9ee73-5b1b-4db3-8fa3-c6202c347e52.png)

<font style="color:rgb(25, 27, 31);">上图 Decoder 接收了 Encoder 的编码矩阵</font>**<font style="color:rgb(25, 27, 31);"> C</font>**<font style="color:rgb(25, 27, 31);">，然后首先输入一个翻译开始符 "<Begin>"，预测第一个单词 "I"；然后输入翻译开始符 "<Begin>" 和单词 "I"，预测单词 "have"，以此类推。</font>

### <font style="color:rgb(15, 17, 21);">关键公式与解释</font>
#### <font style="color:rgb(15, 17, 21);">公式 (1)：Scaled Dot-Product Attention</font>
![](https://cdn.nlark.com/yuque/0/2025/png/56086314/1762755942642-8ad04e86-0949-48f8-96d3-957c33ff03a9.png)

![image](https://cdn.nlark.com/yuque/__latex/946070a2b93c65ac340c3b2b95789033.svg)

| <font style="color:rgb(15, 17, 21);">符号</font> | <font style="color:rgb(15, 17, 21);">含义</font> |
| --- | --- |
| <font style="color:rgb(15, 17, 21);"></font>_<font style="color:rgb(15, 17, 21);">Q</font>_ | <font style="color:rgb(15, 17, 21);">查询矩阵</font> |
| <font style="color:rgb(15, 17, 21);"></font>_<font style="color:rgb(15, 17, 21);">K</font>_ | <font style="color:rgb(15, 17, 21);">键矩阵</font> |
| <font style="color:rgb(15, 17, 21);"></font>_<font style="color:rgb(15, 17, 21);">V</font>_ | <font style="color:rgb(15, 17, 21);">值矩阵</font> |
| <font style="color:rgb(15, 17, 21);"></font>_<font style="color:rgb(15, 17, 21);">dk</font>_ | <font style="color:rgb(15, 17, 21);">键维度</font> |
| <font style="color:rgb(15, 17, 21);"></font>_<font style="color:rgb(15, 17, 21);">dk</font>_ | <font style="color:rgb(15, 17, 21);">缩放因子，防止梯度消失</font> |


##### <font style="color:rgb(25, 27, 31);">Q, K, V 的计算</font>
<font style="color:rgb(25, 27, 31);">Self-Attention 的输入用矩阵X进行表示，则可以使用线性变阵矩阵</font>**<font style="color:rgb(25, 27, 31);">WQ,WK,WV</font>**<font style="color:rgb(25, 27, 31);">计算得到</font>**<font style="color:rgb(25, 27, 31);">Q,K,V</font>**<font style="color:rgb(25, 27, 31);">。计算如下图所示，</font>**<font style="color:rgb(25, 27, 31);">注意 X, Q, K, V 的每一行都表示一个单词。</font>**

![](https://cdn.nlark.com/yuque/0/2025/png/56086314/1762756131293-80eaa69d-2524-40a0-ac58-e0601ac9605c.png)

##### <font style="color:rgb(25, 27, 31);">Self-Attention 的输出</font>
<font style="color:rgb(25, 27, 31);">得到矩阵 Q, K, V之后就可以计算出 Self-Attention 了，公式如下：</font>

![image](https://cdn.nlark.com/yuque/__latex/946070a2b93c65ac340c3b2b95789033.svg)

<font style="color:rgb(15, 17, 21);">为了防止点积结果过大，将注意力分数推入Softmax函数的梯度饱和区，从而导致梯度消失、模型训练不稳定。所以需要除</font>![image](https://cdn.nlark.com/yuque/__latex/ab1300d712e9e97c9e2657c79b9c40a0.svg)<font style="color:rgb(15, 17, 21);">。</font>**<font style="color:rgb(25, 27, 31);">Q</font>**<font style="color:rgb(25, 27, 31);">乘以</font>**<font style="color:rgb(25, 27, 31);">K</font>**<font style="color:rgb(25, 27, 31);">的转置后，得到的矩阵行列数都为 n，n 为句子单词数，这个矩阵可以表示单词之间的 attention 强度。下图为</font>**<font style="color:rgb(25, 27, 31);">Q</font>**<font style="color:rgb(25, 27, 31);">乘以 </font>![image](https://cdn.nlark.com/yuque/__latex/703d6d1833d8fc879a427eb0f3da358e.svg)<font style="color:rgb(25, 27, 31);"> ，1234 表示的是句子中的单词。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/56086314/1762756728579-71ac0a96-28d5-4705-ad6e-3b3f12c9cd9b.png)

得到![image](https://cdn.nlark.com/yuque/__latex/dc43c7c234142f02ae35b8b2c9578e7c.svg)之后，使用 Softmax 计算每一个单词对于其他单词的 attention 系数，公式中的 Softmax 是对矩阵的每一行进行 Softmax，即每一行的和都变为 1.

![](https://cdn.nlark.com/yuque/0/2025/png/56086314/1762756791781-b3a0611d-0ff4-486a-8331-69094c1c2e26.png)

<font style="color:rgb(25, 27, 31);">得到 Softmax 矩阵之后可以和</font>**<font style="color:rgb(25, 27, 31);">V</font>**<font style="color:rgb(25, 27, 31);">相乘，得到最终的输出</font>**<font style="color:rgb(25, 27, 31);">Z</font>**<font style="color:rgb(25, 27, 31);">。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/56086314/1762756808723-f8004bab-a38f-4e9b-a438-10873a420e8b.png)

<font style="color:rgb(25, 27, 31);">上图中 Softmax 矩阵的第 1 行表示单词 1 与其他所有单词的 attention 系数，最终单词 1 的输出 </font>![image](https://cdn.nlark.com/yuque/__latex/4774d09df340b823f29e88c62209f69e.svg)<font style="color:rgb(25, 27, 31);"> 等于所有单词 i 的值 </font>![image](https://cdn.nlark.com/yuque/__latex/588455b04479646f6b7a4a2886b01dba.svg)<font style="color:rgb(25, 27, 31);"> 根据 attention 系数的比例加在一起得到，如下图所示：</font>

![](https://cdn.nlark.com/yuque/0/2025/png/56086314/1762756907676-441dfcb2-84b0-4af5-9169-8f38e3142dbe.png)

#### <font style="color:rgb(15, 17, 21);">公式 (2)：Multi-Head Attention</font>
<font style="color:rgb(25, 27, 31);">Multi-Head Attention 是由多个 Self-Attention 组合形成的，下图是论文中 Multi-Head Attention 的结构图。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/56086314/1762757879802-9617b8a8-c5d4-47f7-816f-f708419750e8.png)

<font style="color:rgb(25, 27, 31);">从上图可以看到 Multi-Head Attention 包含多个 Self-Attention 层，首先将输入</font>**<font style="color:rgb(25, 27, 31);">X</font>**<font style="color:rgb(25, 27, 31);">分别传递到 h 个不同的 Self-Attention 中，计算得到 h 个输出矩阵</font>**<font style="color:rgb(25, 27, 31);">Z</font>**<font style="color:rgb(25, 27, 31);">。下图是 h=8 时候的情况，此时会得到 8 个输出矩阵</font>**<font style="color:rgb(25, 27, 31);">Z</font>**<font style="color:rgb(25, 27, 31);">。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/56086314/1762758028047-8c9f9802-f7c4-424f-b852-fc1e49e23158.png)

![image](https://cdn.nlark.com/yuque/__latex/34ce9c6df7c63fbd1ee1970719c5ff46.svg)

![image](https://cdn.nlark.com/yuque/__latex/60cf9a2661c3be3229e570ac2c0a56c1.svg)

得到 8 个输出矩阵 ![image](https://cdn.nlark.com/yuque/__latex/4774d09df340b823f29e88c62209f69e.svg) 到 ![image](https://cdn.nlark.com/yuque/__latex/af1c87a76e58acf9ae61f81a3cc839c1.svg)之后，Multi-Head Attention 将它们拼接在一起 (Concat)，然后传入一个Linear层，得到 Multi-Head Attention 最终的输出Z。

![](https://cdn.nlark.com/yuque/0/2025/png/56086314/1762758136285-dc8ea773-d58e-4961-a6b8-45475607bce9.png)

<font style="color:rgb(25, 27, 31);">可以看到 Multi-Head Attention 输出的矩阵</font>**<font style="color:rgb(25, 27, 31);">Z</font>**<font style="color:rgb(25, 27, 31);">与其输入的矩阵</font>**<font style="color:rgb(25, 27, 31);">X</font>**<font style="color:rgb(25, 27, 31);">的维度是一样的。</font>

#### <font style="color:rgb(15, 17, 21);">公式 (3)：位置编码（正弦/余弦）</font>
![image](https://cdn.nlark.com/yuque/__latex/dbe34879596bf31bf1a117c56514abf0.svg)

![image](https://cdn.nlark.com/yuque/__latex/e009867d1fdab4bdcdd049ba91de7a98.svg)

<font style="color:rgb(25, 27, 31);">位置 Embedding 用 </font>**<font style="color:rgb(25, 27, 31);">PE</font>**<font style="color:rgb(25, 27, 31);">表示, pos 表示单词在句子中的位置，d 表示 PE的维度 (与词 Embedding 一样)，2i 表示偶数的维度，2i+1 表示奇数维度 (即 2i≤d, 2i+1≤d)。使用这种公式计算 PE 有以下的好处：</font>

1. **<font style="color:rgb(15, 17, 21);">能够泛化到更长的序列</font>**<font style="color:rgb(15, 17, 21);">： 由于正弦函数的性质</font><font style="color:rgb(15, 17, 21);"> </font>`<font style="color:rgb(15, 17, 21);background-color:rgb(235, 238, 242);">sin(a+b) = sin(a)cos(b) + cos(a)sin(b)</font>`<font style="color:rgb(15, 17, 21);">，模型有可能学会关注</font>**<font style="color:rgb(15, 17, 21);">相对位置</font>**<font style="color:rgb(15, 17, 21);">。例如，对于偏移量</font><font style="color:rgb(15, 17, 21);"> </font>`<font style="color:rgb(15, 17, 21);background-color:rgb(235, 238, 242);">k</font>`<font style="color:rgb(15, 17, 21);">，</font>`<font style="color:rgb(15, 17, 21);background-color:rgb(235, 238, 242);">PE(pos+k)</font>`<font style="color:rgb(15, 17, 21);"> </font><font style="color:rgb(15, 17, 21);">可以表示为</font><font style="color:rgb(15, 17, 21);"> </font>`<font style="color:rgb(15, 17, 21);background-color:rgb(235, 238, 242);">PE(pos)</font>`<font style="color:rgb(15, 17, 21);"> </font><font style="color:rgb(15, 17, 21);">的线性函数。这意味着，即使模型在训练时从未见过长度为100的序列，它也可能在一定程度上推断出位置100和位置50的相对关系。</font>
2. **<font style="color:rgb(15, 17, 21);">确定性且不需要额外参数</font>**<font style="color:rgb(15, 17, 21);">： 它是通过公式计算的，不需要像可学习的位置嵌入那样增加模型参数，并且对于任何长度的序列都是确定的。</font>

<font style="color:rgb(25, 27, 31);"></font>

**<font style="color:rgb(25, 27, 31);">补充：</font>**

<font style="color:rgb(15, 17, 21);">自注意力机制就像一个‘超级词袋模型’。它能看到所有的词，并计算它们之间的关系，但它有一个本质缺陷：</font>**<font style="color:rgb(15, 17, 21);">它对位置信息是完全不敏感的</font>**<font style="color:rgb(15, 17, 21);">。</font>

<font style="color:rgb(15, 17, 21);">举个例子，‘猫抓老鼠’和‘老鼠抓猫’，这两句话的语义天差地别。但对于自注意力来说，它看到的只是‘猫’、‘抓’、‘老鼠’这三个词，以及它们两两之间的关联强度。</font>**<font style="color:rgb(15, 17, 21);">它完全不知道哪个词在前，哪个词在后。</font>**

<font style="color:rgb(15, 17, 21);">这种性质在学术上被称为‘置换不变性’。打乱输入顺序，只会打乱输出顺序，但不会改变每个输出位置所聚合的‘信息内容’。这显然无法满足我们对语言的理解需求。</font>

<font style="color:rgb(15, 17, 21);">Transformer的解决方案：显式地添加位置编码</font>

<font style="color:rgb(15, 17, 21);"></font>

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




