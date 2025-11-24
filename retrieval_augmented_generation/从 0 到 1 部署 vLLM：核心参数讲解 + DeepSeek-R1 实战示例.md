## 1. vLLM 是什么？我们要部署成什么样子？
[vLLM](https://github.com/vllm-project/vllm) 是一个专门做 **大模型推理 & 服务** 的引擎，特点是：

+ 内部用到了 **PagedAttention** 等技术，把 KV Cache 当“分页内存”管理，提高显存利用率和吞吐；([VLLM Docs](https://docs.vllm.ai/en/latest/getting_started/quickstart/?utm_source=chatgpt.com))
+ 提供一个 **OpenAI 兼容的 HTTP 服务**，支持 Chat / Completions / Embeddings 等接口，可以直接用 OpenAI 官方的 Python SDK 来访问本地模型。([VLLM Docs](https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html?utm_source=chatgpt.com))

官方推荐的启动方式是：

```bash
# 新版 vLLM 推荐用 vllm serve
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --api-key your-key \
  --port 8000
```

更底层的方式：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --api-key your-key \
  --port 8000
```

两种本质是同一套 OpenAI 兼容服务，只是入口不一样。([nm-vllm.readthedocs.io](https://nm-vllm.readthedocs.io/en/0.5.0/serving/openai_compatible_server.html?utm_source=chatgpt.com))

---

## 2. vLLM 部署的整体步骤
无论你部署哪个模型，大体流程都一样：

1. **准备环境**
    - 安装显卡驱动 / CUDA；
    - 创建 Python 虚拟环境；
    - 安装 `vllm` + `openai` 等依赖。
2. **准备模型**
    - 要么写 HuggingFace Hub 名字（例如 `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`）；
    - 要么事先下载到本地，然后指定本地路径。([ploomber.io](https://ploomber.io/blog/vllm-deploy/?utm_source=chatgpt.com))
3. **启动 vLLM OpenAI 兼容服务**
    - 选择 `--model`、`--host`、`--port`、`--api-key` 等基本参数；
    - 根据硬件情况调 `--gpu-memory-utilization`、`--max-model-len` 等资源相关参数。
4. **客户端用 OpenAI 协议调用**
    - `base_url` 指向你自己的 vLLM 服务（例如 `http://localhost:9102/v1`）；
    - `api_key` 写成你启动服务时设置的那个值；
    - `model` 字段用 `--served-model-name` 指定的名字。

---

## 3. vLLM 部署必须掌握的几类参数
vLLM 的参数很多，官方文档里分为 **Engine Arguments / Server Arguments** 等分类。([VLLM Docs](https://docs.vllm.ai/en/latest/configuration/engine_args/?utm_source=chatgpt.com))  
常用参数如下。

### 3.1 模型相关参数
#### `--model`
+ **作用**：指定要加载的模型，可以是 HF 名字，也可以是本地路径。([nm-vllm.readthedocs.io](https://nm-vllm.readthedocs.io/en/0.5.0/serving/openai_compatible_server.html?utm_source=chatgpt.com))
+ 示例：

```bash
# 从 HuggingFace 拉取
--model deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# 从本地目录加载
--model /data/models/DeepSeek-R1-Distill-Llama-8B
```

> 提示：从 HF 拉取时，有些私有模型需要事先设置 `HF_TOKEN` 环境变量，否则会报找不到仓库。([ploomber.io](https://ploomber.io/blog/vllm-deploy/?utm_source=chatgpt.com))
>

#### `--dtype`
+ **作用**：指定推理时的数值精度，如 `float16` / `bfloat16` / `auto` 等。([nm-vllm.readthedocs.io](https://nm-vllm.readthedocs.io/en/0.5.0/serving/openai_compatible_server.html?utm_source=chatgpt.com))
+ 常见用法：

```bash
--dtype auto        # 让 vLLM 自动选择
--dtype float16     # 强制 fp16
```

如果只是简单部署，不确定选啥，`--dtype auto`** 一般就够用**。

#### `--chat-template`
+ **作用**：指定 chat 模型的 prompt 模板（Jinja 格式），当模型 config 里没有自带 chat template 时，必须手动指定，否则 chat 请求会报错。([nm-vllm.readthedocs.io](https://nm-vllm.readthedocs.io/en/0.4.0/serving/openai_compatible_server.html?utm_source=chatgpt.com))
+ 示例：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model your-model \
  --chat-template ./path/to/template.jinja
```

对于像 DeepSeek-R1 这类已有官方 chat 模板的模型，一般**可以不手动写**，但如果发现 `/v1/chat/completions` 报错，就往这个方向排查。

---

### 3.2 服务相关参数（OpenAI 这一层）
这些参数决定了“你的服务在网络上长什么样”。

#### `--host` & `--port`
+ `--host`：监听地址
    - `127.0.0.1`：只本机访问；
    - `0.0.0.0`：对所有网卡暴露（常见于服务器部署）。([VLLM Docs](https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html?utm_source=chatgpt.com))
+ `--port`：监听端口
    - 随便选个不冲突的端口，比如 `8000 / 9102 / 6006` 等。

```bash
--host 0.0.0.0 \
--port 9102
```

#### `--api-key`
+ **作用**：OpenAI 兼容接口用的 API Key。([nm-vllm.readthedocs.io](https://nm-vllm.readthedocs.io/en/0.5.0/serving/openai_compatible_server.html?utm_source=chatgpt.com))
+ 实际效果是：服务端启动时要求请求 header 带 `Authorization: Bearer <API_KEY>`，否则就 401。

```bash
--api-key my-super-secret-key
```

客户端调用时：

```bash
-H "Authorization: Bearer my-super-secret-key"
```

#### `--served-model-name`
+ **作用**：对外暴露的模型名，就是客户端 `model` 字段写的那个值。([nm-vllm.readthedocs.io](https://nm-vllm.readthedocs.io/en/0.4.0/serving/openai_compatible_server.html?utm_source=chatgpt.com))
+ 如果不指定，默认等于 `--model` 的值（比如 HF 名字或者本地路径，这对客户端不太友好）。

```bash
--model /data/models/DeepSeek-R1-Distill-Llama-8B \
--served-model-name deepseek-r1-llama8b
```

这样客户端就可以写：

```json
{ "model": "deepseek-r1-llama8b", ... }
```

而不需要关心你究竟是从 HF 拉的，还是从什么路径加载的。([GitHub](https://github.com/vllm-project/vllm/issues/13257?utm_source=chatgpt.com))

---

### 3.3 GPU / 显存 / 并发相关参数
这部分是 vLLM 调优的重点，官方和社区文章也都围绕这几个参数展开。([VLLM Docs](https://docs.vllm.ai/en/latest/configuration/optimization/?utm_source=chatgpt.com))

#### `--gpu-memory-utilization`
+ **作用**：控制 vLLM 能用的 GPU 显存比例（0.0–1.0）。([VLLM Docs](https://docs.vllm.ai/en/v0.7.2/serving/engine_args.html?utm_source=chatgpt.com))
+ 默认值通常是 **0.9（90%）**，推荐在 `0.9–0.95` 之间调节，避免 OOM。([VLLM Docs](https://docs.vllm.ai/en/latest/configuration/optimization/?utm_source=chatgpt.com))
+ 如果你要在同一张卡上跑多个 vLLM 实例，可以为每个实例设成 0.4/0.5 之类。([Red Hat 文档](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.0/html/vllm_server_arguments/all-server-arguments-server-arguments?utm_source=chatgpt.com))

示例：

```bash
--gpu-memory-utilization 0.90
```

#### `--max-model-len`
+ **作用**：限制“模型最大上下文长度（token 数）”。([Red Hat 文档](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.0/html/vllm_server_arguments/key-server-arguments-server-arguments?utm_source=chatgpt.com))
+ 如果模型原生上下文很长（比如 16k/32k），但你的应用最多只用 4k/8k，可以**主动设小一点**，减少显存占用、降低 OOM 概率。([Towards AI](https://pub.towardsai.net/vllm-parameters-tuning-for-better-performance-f4014b50e09c?utm_source=chatgpt.com))

示例：

```bash
--max-model-len 8192
```

简单经验：

+ 日常聊天 / 办公场景：`4096 ~ 8192` 就够用；
+ 真要长文处理再单独起一个“长上下文实例”。

#### `--max-num-batched-tokens`
+ **作用**：一次步进中，所有请求加起来的 **最大 token 数**，越大吞吐越高，但显存占用和首 token 延迟也会增加。([Red Hat 文档](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.0/html/vllm_server_arguments/key-server-arguments-server-arguments?utm_source=chatgpt.com))

```bash
--max-num-batched-tokens 32768
```

一般建议从 `8192 ~ 32768` 这个量级试起，然后根据延迟和显存表现做小幅调整。([ROCm Documentation](https://rocm.docs.amd.com/en/docs-7.1.0/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html?utm_source=chatgpt.com))

#### `--max-num-seqs`
+ **作用**：一次调度中允许的并发“序列”数量上限。
+ 当你有大量并发请求时，适当增大这个值可以提升吞吐，但 GPU 要扛得住。

```bash
--max-num-seqs 512
```

这几个参数**是相互耦合的**：`max-model-len * max-num-seqs` 再乘上一个常数，大致就决定了 KV Cache 占用；`max-num-batched-tokens` 又决定了单步能塞进去多少 token。调整时建议从保守的组合开始，逐步放宽。([VLLM Docs](https://docs.vllm.ai/en/latest/configuration/optimization/?utm_source=chatgpt.com))

#### `--tensor-parallel-size`（多卡）
+ **作用**：张量并行，把一个模型拆分到多张 GPU 上。([Red Hat 文档](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.0/html/vllm_server_arguments/key-server-arguments-server-arguments?utm_source=chatgpt.com))

示例（两张卡）：

```bash
--tensor-parallel-size 2
```

在单卡就不用加这个参数了。

---

## 4. 常用参数小抄（速查表）
> 不是完整列表，只列出个人感觉 **“你 90% 的时候会改到的那几个”**。
>

| 类别 | 参数 | 含义（简化） |
| --- | --- | --- |
| 模型 | `--model` | HF 名字或本地路径 |
| 模型 | `--dtype` | 推理精度，例如 `auto` / `float16` |
| 模型 | `--chat-template` | 手工提供 chat 模板 |
| 服务 | `--host` / `--port` | 服务监听地址 / 端口 |
| 服务 | `--api-key` | OpenAI 兼容接口的 API Key |
| 服务 | `--served-model-name` | 对外暴露的模型名（客户端 `model=`） |
| 资源 | `--gpu-memory-utilization` | vLLM 占用 GPU 显存比例，默认 0.9 |
| 资源 | `--max-model-len` | 最大上下文长度（token） |
| 资源 | `--max-num-batched-tokens` | 单步最多 batch 的 token 总数 |
| 资源 | `--max-num-seqs` | 单步最多同时服务的序列数 |
| 资源 | `--tensor-parallel-size` | 多卡张量并行（单卡不用） |


---

## 5. 实战：在一张 4090 上用 vLLM 部署 DeepSeek-R1-Distill-Llama-8B
**可直接运行的一键脚本**，放在例如 `scripts/start_deepseek_r1_llama8b.sh` 里。

> 假设前提：
>
> + 你已经安装好 vLLM 和相关依赖；
> + 模型已经下载到本地某个目录。
>

```bash
#!/usr/bin/env bash
set -e

########################################
# 基本配置
########################################

# 只用第 0 张 4090，保留第 1 张给 2B embedding 模型
export CUDA_VISIBLE_DEVICES=1

# 模型本地路径（按你本地的实际路径改）
MODEL_NAME="/home/ubuntu/data/yuafowo/workspace/Dora-Smart-Display/model_zoo/DeepSeek-R1-Distill-Llama-8B"

# 服务配置
PORT=9102                              # 换一个不冲突的端口
API_KEY="deepseek-r1-llama8b-key"      # 换成你自己的 key
LOG_FILE="deepseek-r1-llama8b.log"     # 日志文件名
HOST="0.0.0.0"
SERVED_NAME="deepseek-r1-llama8b"      # 对外暴露的模型名

########################################
# 启动信息
########################################
echo "Starting DeepSeek-R1-Distill-Llama-8B server..."
echo "  Model: ${MODEL_NAME}"
echo "  Host:  ${HOST}"
echo "  Port:  ${PORT}"
echo "  Log:   ${LOG_FILE}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

########################################
# 后台启动 vLLM OpenAI API Server
########################################
nohup python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --served-model-name "${SERVED_NAME}" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "DeepSeek-R1-Distill-Llama-8B server started with PID ${PID}"
echo "Logs are being written to ${LOG_FILE}"
```

### 5.1 如何运行
```bash
chmod +x scripts/start_deepseek_r1_llama8b.sh
./scripts/start_deepseek_r1_llama8b.sh

# 看下日志
tail -f deepseek-r1-llama8b.log
```

看到类似：

```latex
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9102
```

说明服务已经跑起来了。

---

## 6. 示例脚本逐段说明（通用思路）
这个脚本也可以看成一个“**vLLM 通用启动模板**”，换模型时只需要改几行。

### 6.1 GPU 选择：`CUDA_VISIBLE_DEVICES`
```bash
export CUDA_VISIBLE_DEVICES=1
```

+ 这句的意思是：**只让当前进程看到“物理 1 号卡”**；
+ 对这个进程来说，它看到的卡会被重新编号成 0，所以你可以理解为“给这个服务单独划了一张卡”。

如果你希望：

+ LLM 用 0 号卡：`export CUDA_VISIBLE_DEVICES=0`；
+ embedding 用 1 号卡：在另一个进程里 `export CUDA_VISIBLE_DEVICES=1`。

### 6.2 模型路径：`MODEL_NAME`
```bash
MODEL_NAME="/home/ubuntu/.../DeepSeek-R1-Distill-Llama-8B"
```

+ 这里写的是 **本地模型目录**；
+ 如果你想从 HF 拉，也可以直接写 `MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"`，vLLM 会自动下载。([nm-vllm.readthedocs.io](https://nm-vllm.readthedocs.io/en/0.5.0/serving/openai_compatible_server.html?utm_source=chatgpt.com))

### 6.3 服务配置：端口 / Host / API Key / 模型名
```bash
PORT=9102
API_KEY="deepseek-r1-llama8b-key"
LOG_FILE="deepseek-r1-llama8b.log"
HOST="0.0.0.0"
SERVED_NAME="deepseek-r1-llama8b"
```

+ `PORT`：换成你机器上没被占用的端口即可；
+ `API_KEY`：随便设个复杂一点的字符串，客户端要一致；
+ `HOST`：如果只是本机开发，可以改成 `127.0.0.1`；
+ `SERVED_NAME`：推荐用一个短而好记的名字，客户端 `model` 字段就写这个。([nm-vllm.readthedocs.io](https://nm-vllm.readthedocs.io/en/0.4.0/serving/openai_compatible_server.html?utm_source=chatgpt.com))

### 6.4 vLLM 参数：显存 & 上下文
```bash
--gpu-memory-utilization 0.90 \
--max-model-len 8192 \
```

+ `--gpu-memory-utilization 0.90`
    - 预留 90% 显存给权重 + KV Cache；
    - 如果你在同一张卡上还要跑其他服务，可以降到 0.8 甚至更低。([VLLM Docs](https://docs.vllm.ai/en/latest/configuration/optimization/?utm_source=chatgpt.com))
+ `--max-model-len 8192`
    - 对于 DeepSeek-R1-8B，这个设置在“够用 + 不太容易 OOM”的中间位置；
    - 如果你的场景主要是短对话，可以改成 `4096` 换一点显存空间出来。([Red Hat 文档](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.0/html/vllm_server_arguments/key-server-arguments-server-arguments?utm_source=chatgpt.com))

你可以在这个脚本的基础上继续加：

```bash
  --max-num-batched-tokens 32768 \
  --max-num-seqs 512 \
```

来调优吞吐量和并发。

---

## 7. 客户端如何调用这个服务？
### 7.1 Python（OpenAI SDK）
```python
from openai import OpenAI

client = OpenAI(
    api_key="deepseek-r1-llama8b-key",          # 和脚本一致
    base_url="http://127.0.0.1:9102/v1",        # 注意要带 /v1
)

resp = client.chat.completions.create(
    model="deepseek-r1-llama8b",                # 对应 --served-model-name
    messages=[
        {"role": "user", "content": "用三句话介绍一下 DeepSeek-R1-Distill-Llama-8B。"}
    ],
    max_tokens=512,
    temperature=0.7,
)

print(resp.choices[0].message.content)
```

vLLM 的 OpenAI 兼容服务就是挂在 `/v1` 之下的，所以 `base_url` 通常写成 `http://host:port/v1`。 ([VLLM Docs](https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html?utm_source=chatgpt.com))

### 7.2 curl
```bash
curl -X POST "http://127.0.0.1:9102/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer deepseek-r1-llama8b-key" \
  -d '{
    "model": "deepseek-r1-llama8b",
    "messages": [
      { "role": "user", "content": "你好，做个自我介绍。" }
    ]
  }'
```

---



