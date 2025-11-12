[toc]

## 前言
BERT是一种深度双向预训练模型，通过掩蔽语言模型（MLM）和下一句预测（NSP）任务进行预训练，显著提高了在多个自然语言处理任务中的表现，尤其在问答和自然语言推理任务中取得了新的最先进成绩。

论文链接：[https://arxiv.org/pdf/1810.04805](https://arxiv.org/pdf/1810.04805)

![](https://cdn.nlark.com/yuque/0/2025/png/56086314/1762935606169-ddc1ef9a-9d8e-43df-88e0-98e088ac22f8.png)

**BERT的整体预训练和微调过程。除了输出层外，预训练和微调使用的是相同的架构。相同的预训练模型参数用于初始化不同下游任务的模型。在微调过程中，所有的参数都会进行微调。[CLS]是一个特殊符号，添加在每个输入示例的前面，[SEP]是一个特殊的分隔符（例如，用于分隔问题和答案）。**

## 研究背景
### 问题定义与动机
自然语言处理（NLP）中的预训练语言模型已经被证明能有效提升多种任务的表现。传统的模型如OpenAI GPT采用单向语言模型（左到右），而现有的方法如ELMo结合了双向信息，但仍没有做到深度的双向编码。当前技术的局限性在于：<font style="color:#DF2A3F;">语言模型的上下文依赖性过于简单，难以有效捕捉句子之间的细节关系。</font>

### 该问题的重要性
NLP任务的复杂性要求模型能够充分理解语境，而现有的模型架构通常受到“单向”或“浅层双向”限制。BERT通过引入深度双向Transformer架构解决了这一问题，能够在多种任务上提供更准确的预测，尤其是在问答和自然语言推理等需要深度语境理解的任务上。

## 方法细节
### 核心思想
BERT通过使用掩蔽语言模型（MLM）和下一句预测（NSP）任务，使得预训练过程中模型能够同时依赖左右两侧的上下文信息，解决了传统模型的单向限制。

### 整体流程
1. **预训练阶段：**
    - 使用掩蔽语言模型（MLM）任务来训练双向Transformer，使模型能够根据上下文信息预测缺失的词汇。
    - 使用下一句预测（NSP）任务来捕捉句子对之间的关系，提升模型在句子级任务中的表现。
2. **微调阶段：**
    - 采用统一的模型架构，微调时通过简单的额外输出层适应具体任务。

### 关键模块
+ **Masked Language Model (MLM)**：通过随机掩蔽输入的部分单词，强迫模型利用完整上下文来预测掩蔽单词。
+ **Next Sentence Prediction (NSP)**：通过训练模型预测两个句子是否是连续的，帮助模型理解句子对的关系。

### 伪代码
```python
def pretrain_BERT():
    for each_batch in dataset:
        masked_tokens = mask_random_tokens(each_batch)
        predict(masked_tokens)  # 使用掩蔽语言模型预测掩蔽词
        next_sentence_prob = predict_next_sentence(each_batch)  # 下一句预测
```

### 关键公式：
1. **MLM 公式：**

![image](https://cdn.nlark.com/yuque/__latex/f2b8bd104966bb1ba64c814aebb9529b.svg)

其中 ![image](https://cdn.nlark.com/yuque/__latex/d99fd2df7b5f652a4b7fc593fb9df750.svg) 是被掩蔽的词汇，模型需要基于上下文来预测其真实词汇。<font style="color:rgba(0, 0, 0, 0.85);">BERT 的 MLM 任务是</font>**<font style="color:rgb(0, 0, 0) !important;">随机掩盖输入句子中的部分单词，让模型根据上下文预测这些被掩盖的单词</font>**<font style="color:rgba(0, 0, 0, 0.85);">。这个损失函数的目标是：让模型对 “正确单词</font>![image](https://cdn.nlark.com/yuque/__latex/d99fd2df7b5f652a4b7fc593fb9df750.svg)<font style="color:rgba(0, 0, 0, 0.85);">” 的预测概率尽可能大，从而最小化整体损失。</font>

    - ![image](https://cdn.nlark.com/yuque/__latex/e7e8b8e93598ae8e762ab96e902f7359.svg)<font style="color:rgb(0, 0, 0);">：表示掩码语言模型的损失函数，用于衡量模型在 “预测被掩码的单词” 任务上的误差。</font>
    - ![image](https://cdn.nlark.com/yuque/__latex/6c838f12793f42d989bddaf71c52439b.svg)<font style="color:rgb(0, 0, 0);">：</font>
        * <font style="color:rgb(0, 0, 0);">求和符号</font><font style="color:rgb(0, 0, 0);">∑</font><font style="color:rgb(0, 0, 0);">表示对所有被掩码的目标词进行 “损失计算” 的累加；</font>
        * <font style="color:rgb(0, 0, 0);">负号 “</font><font style="color:rgb(0, 0, 0);">−</font><font style="color:rgb(0, 0, 0);">” 是为了将</font>**<font style="color:rgb(0, 0, 0) !important;">对数似然的最大化问题</font>**<font style="color:rgb(0, 0, 0);">转换为</font>**<font style="color:rgb(0, 0, 0) !important;">损失函数的最小化问题</font>**<font style="color:rgb(0, 0, 0);">（机器学习中通常以 “最小化损失” 为优化目标）；</font>
        * _<font style="color:rgb(0, 0, 0);">i</font>_<font style="color:rgb(0, 0, 0);">=</font><font style="color:rgb(0, 0, 0);">1</font><font style="color:rgb(0, 0, 0);">到</font>_<font style="color:rgb(0, 0, 0);">N</font>_<font style="color:rgb(0, 0, 0);">表示共有</font>_<font style="color:rgb(0, 0, 0);">N</font>_<font style="color:rgb(0, 0, 0);">个被掩码的目标词需要预测。</font>
    - ![image](https://cdn.nlark.com/yuque/__latex/d01fc6f01d39511b9799907efc21a988.svg)<font style="color:rgba(0, 0, 0, 0.85);">：</font>
        * ![image](https://cdn.nlark.com/yuque/__latex/2b44ab25821746410f89e098e511ac8e.svg)<font style="color:rgb(0, 0, 0);">表示 “在给定上下文</font>_<font style="color:rgb(0, 0, 0);">context</font>_<font style="color:rgb(0, 0, 0);">的情况下，模型预测单词</font>![image](https://cdn.nlark.com/yuque/__latex/d99fd2df7b5f652a4b7fc593fb9df750.svg)<font style="color:rgb(0, 0, 0);">的概率”；</font>
        * <font style="color:rgb(0, 0, 0);">log（对数）是为了将 “概率的连乘” 转换为 “对数的累加”，避免数值下溢，同时保证优化的平滑性。	</font>
2. **NSP 公式：**

![image](https://cdn.nlark.com/yuque/__latex/05fdd53bcf781a145fadfd516dc4ffef.svg)

<font style="color:rgb(0, 0, 0);"></font>

<font style="color:rgb(0, 0, 0);">在预训练时，BERT 会构造两类样本：</font>

+ **<font style="color:rgb(0, 0, 0) !important;">正样本</font>**<font style="color:rgb(0, 0, 0);">：句子 B 确实是句子 A 的下一句（比如从同一篇文章中连续截取的两个句子）；</font>
+ **<font style="color:rgb(0, 0, 0) !important;">负样本</font>**<font style="color:rgb(0, 0, 0);">：句子 B 不是句子 A 的下一句（比如从不同文章中随机抽取的两个句子）。</font>

<font style="color:rgb(0, 0, 0);">这个损失函数的目标是：让模型对</font>**<font style="color:rgb(0, 0, 0) !important;">正样本</font>**<font style="color:rgb(0, 0, 0);">输出的</font>_<font style="color:rgb(0, 0, 0);">P</font>_<font style="color:rgb(0, 0, 0);">(IsNext)尽可能接近 1，对</font>**<font style="color:rgb(0, 0, 0) !important;">负样本</font>**<font style="color:rgb(0, 0, 0);">输出的</font>_<font style="color:rgb(0, 0, 0);">P</font>_<font style="color:rgb(0, 0, 0);">(IsNext)尽可能接近 0，从而最小化整体损失。</font>

+ ![image](https://cdn.nlark.com/yuque/__latex/400208ff68d2ddb8d36b411fbfb9ddea.svg)<font style="color:rgb(0, 0, 0);">：表示 “下一句预测任务” 的损失函数，用于衡量模型在该任务上的预测误差。</font>
+ ![image](https://cdn.nlark.com/yuque/__latex/100f9bf80d26d06aeacc0ed643611ca4.svg)<font style="color:rgb(0, 0, 0);">：</font>
    - <font style="color:rgb(0, 0, 0);">同 MLM 损失的逻辑，负号将 “最大化正确预测的概率” 转换为 “最小化损失”；</font>
    - <font style="color:rgb(0, 0, 0);">对数</font><font style="color:rgb(0, 0, 0);">lo</font><font style="color:rgb(0, 0, 0);">g</font><font style="color:rgb(0, 0, 0);">用于将概率的 “连乘” 转化为 “累加”，避免数值下溢，同时让优化过程更平滑。</font>
+ ![image](https://cdn.nlark.com/yuque/__latex/689a4e1da0d612a030e98339a35ab064.svg)<font style="color:rgb(0, 0, 0);">：</font>
    - <font style="color:rgb(0, 0, 0);">IsNext</font><font style="color:rgb(0, 0, 0);">是一个二分类标签，表示 “句子 B 是句子 A 的下一句”（标签为 1）或 “不是”（标签为 0）；</font>
    - ![image](https://cdn.nlark.com/yuque/__latex/689a4e1da0d612a030e98339a35ab064.svg)<font style="color:rgb(0, 0, 0);">表示 “在给定句子 A 和句子 B 的情况下，模型预测‘句子 B 是句子 A 下一句’的概率”。</font>

## <font style="color:rgb(0, 0, 0);">实验分析</font>
### 数据集与设置
+ **GLUE任务集**：包含多种NLP任务，如情感分类、自然语言推理、问答等。
+ **SQuAD v1.1 & v2.0**：用于问答任务，模型需要从文章中找出问题的答案。
+ **SWAG**：测试模型的常识推理能力。

| 任务 | BERTBASE | BERTLARGE | OpenAI GPT |
| --- | --- | --- | --- |
| GLUE（平均） | 79.6% | 82.1% | 72.8% |
| SQuAD v1.1 | 88.5% | 90.9% | 85.8% |
| SQuAD v2.0 | 83.1% | 86.3% | 78.0% |


BERT显著超越了现有的NLP基准，尤其在多任务学习和问答任务中表现卓越。



## 总结与讨论
### 贡献点：
1. 提出了基于深度双向Transformer的BERT模型，解决了单向限制问题。
2. 提出了新的预训练任务（MLM和NSP），提高了模型的多任务学习能力。
3. 在多个NLP任务中取得了新的最先进成绩，推动了领域的发展。

### 优势：
1. 高效的多任务学习框架，无需大量任务特定的架构调整。
2. 提供了统一的预训练与微调框架，简化了多种任务的模型应用。
3. 提供了开源代码和预训练模型，便于广泛应用和改进。























