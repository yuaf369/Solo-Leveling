## 总览
典型 RAG 请求的大致延迟组成：

**T_total ≈ T_embed + T_retrieval + T_rerank + T_LLM + T_post + T_network**

+ **T_embed**：对 query 做 embedding 的时间
+ **T_retrieval**：在向量库/搜索引擎里查 top-k 的时间
+ **<font style="color:#DF2A3F;">T_rerank</font>**<font style="color:#DF2A3F;">：用 cross-encoder / heavier 模型重排的时间（数据量小的项目，可以跳过这个流程）</font>
+ **T_LLM**：大模型生成的时间（通常是最大头）
+ **T_post**：模板填充、工具调用、结果拼接
+ **T_network**：API 往返、微服务之间的网络延迟

业界线上交互式 RAG，**可接受的端到端延迟一般是 1–3s**；其中检索和生成各自最好控制在 500–1000ms 以内。

## Embedding & 检索阶段加速
### Embedding侧加速：
策略：小模型 + 缓存

选择更轻的embedding模型，开启对“重复问题/高频query”缓存

qwen3-embedding 0.6b

<font style="color:rgb(0, 0, 0);">nlp_gte_sentence-embedding_chinese-base</font>

### 向量检索加速
策略：ANN 索引 + 适合 RAG 的参数 + 热点缓存

使用 ANN（近似近邻）索引，控制检索参数，降低 `nprobe` / `efSearch` 之类的参数：略微牺牲一点召回换更低延迟。取小一点的 top-k。

冷热分层，高频问答放在 内存型向量库（或 Redis / 内存缓存）。冷数据放在Milvus向量数据库中。

### Rerank阶段
语料数据少，数据准确的情况下，能不用rerank就不用rerank。

如果数据量上来了，数据比较多。使用轻量级模型，并减少候选文档数。

## 大模型生成加速
### 控制token数
控制上下文长度，合理分chunk，不把整篇文章给大模型。

合理控制top-k，控制输出长度。设置合理的max_new_tokens/max_tokens，在prompt里明确指出，字数不能超出X字。

### 端侧推理
使用高性能推理引擎，vllm/TensorRT-LLM等都做了kv cache优化；量化和蒸馏模型，在保证模型精度可以接受的情况下可以通过减少模型权重来降低延迟。用蒸馏的小模型专门来做RAG问答，因为RAG本质上答案来自于检索文档，对模型推理能力要求略低。



## RAG整体加速
### 减少LLM调用次数
对于查询天气这类功能，在大模型底层是使用的function call，会多次调用大模型，所以会变得很慢。（需要深入再次调研）。直接走固定API，让前端调用。

### 减少无效上下文：保证语料库的质量
对于语料数据来说，需要设置更好的chunk策略，不能机械地按照token长度切，需要设置合理的策略。

对于检索后的语料，可以先做轻量过滤（基于规则不基于模型），把不相关的剔除掉。

Prompt压缩 / Context distillation， 精简Prompt，和语料内容





