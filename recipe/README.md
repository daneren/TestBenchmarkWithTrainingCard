# Qwen3 模型训练与评测指南

本项目提供了 Qwen3 系列模型（包括 Qwen3-VL-8B、Qwen3-8B 和 Qwen3-VL-30B-A3B）的训练和评测脚本，支持单机和多机分布式训练。

本文档中的所有实验均基于 H20 实验得到。

## 📋 目录

- [模型训练指标](#模型训练指标)
- [前置要求](#前置要求)
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [模型准备](#模型准备)
- [模型训练](#模型训练)
  - [RL 训练 (Qwen3-VL-8B)](#rl-训练-qwen3-vl-8b)
  - [Qwen3-8B 训练](#qwen3-8b-训练)
  - [Qwen3-VL-30B-A3B-Instruct 训练](#qwen3-vl-30b-a3b-instruct-训练)
- [模型评测](#模型评测)
- [常见问题](#常见问题)

## 📊 模型训练指标

### 📊 Qwen3-VL-8B RL训练 vs 原始模型准确率对比（MMMU-Pro 数据集）

| 模型名称                           | 训练方式      | RL Step | MMMU-Pro 准确率 | 日志文件                                                         |
|------------------------------------|-------------|---------|-----------------|-------------------------------------------------------------------|
| qwen3_vl_8b_orig                   | 原始预训练    | -       | 0.5902          | -                                                                 |
| qwen3_vl_8b_megatron_single_node_60| 单机RL       | 60      | 0.6249          | [qwen3_vl_8b_megatron_single_node.log](./qwen3_vl_8b_megatron_single_node.log) |
| qwen3_vl_8b_megatron_mutil_node_120| 多机RL       | 120     | 0.6231          | [qwen3_vl_8b_megatron_mutil_node.log](./qwen3_vl_8b_megatron_mutil_node.log)   |

**训练效果说明：**
- RL 强化学习训练后，模型在 MMMU-Pro 数据集上的准确率均有明显提升。
    - 单机 RL 训练（step 60）准确率由 0.5902 提升至 0.6249
    - 多机 RL 训练（step 120）准确率由 0.5902 提升至 0.6231




## 🔧 前置要求

- **环境变量配置：**
  - 多机训练需要配置 `MASTER_ADDR`、`MASTER_PORT` 等网络参数
  - 根据实际环境修改 IP 地址和端口号

## ⚠️ 安全警告

**重要：** 文档中的 IP 地址配置仅为示例，请勿直接使用硬编码的 IP 地址！

- **单机训练：** `MASTER_ADDR` 可以设置为 `127.0.0.1` 或本机实际 IP
- **多机训练：** `MASTER_ADDR` 必须设置为实际的主节点 IP，所有节点间必须网络互通
- **网络安全：** 确保训练集群在安全网络环境中运行，避免暴露在公网中

**IP 配置示例：**
- 本机 IP: `127.0.0.1` (单机) 或 `192.168.1.100` (实际 IP)
- 集群主节点: `192.168.1.100`
- 集群工作节点: `192.168.1.101`, `192.168.1.102` 等


## 环境准备

### Step 1: 创建容器

**注意：** 请根据实际情况修改挂载路径和网络配置。

```bash
# 拉取 Docker 镜像
docker pull verlai/verl:vllm011.dev_qwenvl_cp

# 删除已存在的容器（如果存在）
docker rm -f danerli_benchmark

# 创建并启动容器
docker run -tid --entrypoint bash --gpus all --name=danerli_benchmark \
  --privileged --device=/dev/infiniband/ --cap-add=IPC_LOCK \
  --shm-size=500g -w /workspace --network=host \
  -v /mnt/:/mnt/ \
  -v /mnt/cfs/danerli/workspace/test_benchmark/TestBenchmarkWithTrainingCard/:/workspace/ \
  verlai/verl:vllm011.dev_qwenvl_cp

# 进入容器
docker exec -it danerli_benchmark bash
```

### Step 2: 初始化环境

在容器内执行以下命令安装依赖：

```bash
# 配置 pip 镜像源
pip3 config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 安装基础依赖
pip3 install debugpy==1.8.0 

# 安装 verl 框架
pip3 install -e /workspace/verl -i https://mirrors.tencent.com/pypi/simple \
  --extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple

# 安装系统工具
apt-get install -y tmux

# 安装 flash attention（需要重新编译）
pip3 uninstall -y flash_attn && pip install flash_attn==2.5.0 --no-build-isolation

# 安装 megatron-energon
pip install megatron-energon==4.0.0

```

### Step 3: 启动 Ray（多机训练需要）

#### Rank 0 节点（主节点）

在主节点上执行：

```bash
ray stop --force

# 设置环境变量（请根据实际情况修改）
export MASTER_ADDR=<MASTER_NODE_IP>     # 主节点 IP 地址（例如：192.168.1.100）
export MASTER_PORT=23467                # Ray 服务端口
export NODE_SUBADDR_IP=<LOCAL_NODE_IP>  # 当前节点 IP（例如：192.168.1.101）
export GPUS_PER_NODE=8                  # 每个节点的 GPU 数量

# 网络接口配置（根据实际网络环境修改）
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0

# 启动 Ray Head 节点
ray start --head --port=$MASTER_PORT \
  --min-worker-port=20122 \
  --max-worker-port=20999
```

#### Rank 1 节点（工作节点）

在其他节点上执行：

```bash
ray stop --force

# 设置环境变量（与主节点保持一致）
export MASTER_ADDR=<MASTER_NODE_IP>     # 主节点 IP 地址（与 Rank 0 相同）
export MASTER_PORT=23467                # Ray 服务端口（与 Rank 0 相同）
export NODE_SUBADDR_IP=<LOCAL_NODE_IP>  # 当前节点 IP（例如：192.168.1.102）
export GPUS_PER_NODE=8                  # 每个节点的 GPU 数量

# 网络接口配置（根据实际网络环境修改）
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0

# 连接到主节点
ray start --address=$MASTER_ADDR:$MASTER_PORT \
  --min-worker-port=20122 \
  --max-worker-port=20999
```

**提示：**
- 确保所有节点之间网络互通
- `NODE_SUBADDR_IP` 需要设置为当前节点的实际 IP 地址
- 如果使用 InfiniBand，确保 `NCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME` 配置正确


## 📦 数据准备

在容器内执行以下命令下载训练数据：

```bash
# 进入容器（如果还未进入）
docker exec -it danerli_benchmark bash

# 1. 下载 LLM 训练数据
mkdir -vp /workspace/datatsets/qwen-datasets
cd /workspace/datatsets/qwen-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.idx

# 2. 下载 MLM 训练数据（视觉-语言多模态数据）
cd /workspace/Pai-Megatron-Patch/toolkits/multimodal_data_preprocessing
python build_fake_wds_for_vl.py --output-dir /workspace/datatsets/wds
```

**数据说明：**
- LLM 数据：用于语言模型预训练
- MLM 数据：用于视觉-语言多模态模型训练
- 数据下载完成后会保存在 `/workspace/datatsets/` 目录下
- 强化学习的数据已经通过git clone下载到了[../datatsets/geo3k](../datatsets/geo3k/)

## 模型准备

将hf格式的模型转换为mcore格式的模型

```shell

cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen3/run_8xH20.sh \
8B \
/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-8B \
/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-8B-to-mcore  \
false \
true \
bf16

cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen3_vl/run_8xH20.sh \
A3B \
/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-VL-30B-A3B-Instruct  \
/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-VL-30B-A3B-Instruct-to-mcore  \
false \
true \
bf16

```

## 🎯 模型训练

### RL 训练 (Qwen3-VL-8B)

使用强化学习（RL）方法对 Qwen3-VL-8B 模型进行训练。

#### 环境准备

确保已完成 [环境准备](#环境准备)、 [数据准备](#数据准备) 和 [模型准备](#模型准备)步骤。

```bash
docker exec -it danerli_benchmark bash
```

#### 单机训练

适用于单机 8 卡训练：

```bash
# 设置模型路径
export HF_MODEL_PATH=/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-VL-8B-Instruct

# 启动训练（单机）
nnodes=1 bash recipe/qwen3_vl_8b_megatron.sh 2>&1 | tee /workspace/recipe/qwen3_vl_8b_megatron_single_node.log
```

#### 多机训练

适用于多机分布式训练（需要先启动 Ray 服务）：

```bash
# 设置模型路径
export HF_MODEL_PATH=/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-VL-8B-Instruct

# 启动训练（多机，2 节点）
nnodes=2 bash recipe/qwen3_vl_8b_megatron.sh 2>&1 | tee /workspace/recipe/qwen3_vl_8b_megatron_mutil_node.log
```

**参数说明：**
- `HF_MODEL_PATH`: HuggingFace 格式的预训练模型路径
- `nnodes`: 训练节点数量（1=单机，2=双机，以此类推）

#### 训练日志

- **单机运行日志：** [qwen3_vl_8b_megatron_single_node.log](./qwen3_vl_8b_megatron_single_node.log)
- **单机运行wandb日志：** [qwen3_vl_8b_megatron_single_node](https://wandb.ai/lr-experiment/TestBenchmarkWithTrainingCard/runs/cdwvjsde)

- **多机运行日志：** [qwen3_vl_8b_megatron_mutil_node.log](./qwen3_vl_8b_megatron_mutil_node.log)
- **多机运行wandb日志：** [qwen3_vl_8b_megatron_mutil_node](https://wandb.ai/lr-experiment/TestBenchmarkWithTrainingCard/runs/8j9jdbbw)


**训练输出：**
训练完成后，模型 checkpoint 会保存在 `/workspace/recipe/exp/checkpoint/qwen3_vl_8b_megatron_*/` 目录下。


## 📈 模型评测

### 评测环境准备

```shell
docker run -tid --gpus all --privileged --device=/dev/infiniband/ --cap-add=IPC_LOCK --shm-size=500g  --network=host --name=danerli_qwen3vl -v /mnt/:/mnt/ -v/mnt/cfs/danerli/workspace/test_benchmark/TestBenchmarkWithTrainingCard:/workspace -w /workspace qwenllm/qwenvl:qwen3vl-cu128 bash

docker start danerli_qwen3vl && docker exec -it danerli_qwen3vl bash
```

### 安装评测工具

```bash
pip install evalscope==1.2.0
```

### 一键执行脚本

使用 `run_model_evaluation.sh` 脚本可以一键执行模型评测，支持自定义 GPU 号、模型路径和模型名称。


#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `GPU_ID` | GPU 编号，多个 GPU 用逗号分隔 | `0,1,2,3,4,5,6,7` |
| `MODEL_PATH` | 模型路径（HuggingFace 格式） | `/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-VL-8B-Instruct` |
| `MODEL_NAME` | 模型名称（用于结果标识） | `qwen3_vl_8b_orig` |
| `PORT` | vLLM 服务端口 | `8000` |

#### 使用示例

**评测 RL 训练后的模型：**

这三个模型都是大约评测一个小时左右，评测时间过长有可能是模型训练之后模型在评测阶段重复输出，建议检查评测阶段调用模型的repsonse

```bash
# 评测多机训练后的step120保存的模型
export GPU_ID=0,1,2,3,4,5,6,7
export MODEL_PATH=/mnt/cfs/danerli/workspace/test_benchmark/TestBenchmarkWithTrainingCard/recipe/exp/checkpoint/qwen3_vl_8b_megatron_mutil_node/global_step_120/actor/huggingface
export MODEL_NAME=qwen3_vl_8b_megatron_mutil_node_120
export PORT=8000
bash recipe/run_model_evaluation.sh

# 评测单机训练后的step60保存的模型
export GPU_ID=0,1,2,3,4,5,6,7
export MODEL_PATH=/mnt/cfs/danerli/workspace/test_benchmark/TestBenchmarkWithTrainingCard/recipe/exp/checkpoint/qwen3_vl_8b_megatron_/global_step_60/actor/huggingface
export MODEL_NAME=qwen3_vl_8b_megatron_single_node_60
export PORT=8000
bash recipe/run_model_evaluation.sh

# 评测原始模型
export GPU_ID=0,1,2,3,4,5,6,7
export MODEL_PATH=/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-VL-8B-Instruct
export MODEL_NAME=qwen3_vl_8b_orig
export PORT=8000
bash recipe/run_model_evaluation.sh
```

### Qwen3-8B 训练

对 Qwen3-8B 模型进行训练。

#### 环境准备

```bash
docker exec -it danerli_benchmark bash
```

#### 单机训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1    
export MASTER_PORT=23567                # 通信端口
export KUBERNETES_CONTAINER_RESOURCE_GPU=8
export HF_MODEL_PATH=/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-8B-to-mcore

bash /workspace/recipe/run_qwen3_8b_pt.sh 2>&1 | tee /workspace/recipe/run_qwen3_8b_pt_single_node.log
```

#### 多机训练

**Rank 0 节点：**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=2                     # 总节点数
export RANK=0                           # 当前节点 rank
export MASTER_ADDR=<MASTER_NODE_IP>     # 主节点 IP（例如：192.168.1.100）
export MASTER_PORT=23567                # 通信端口
export KUBERNETES_CONTAINER_RESOURCE_GPU=8
export HF_MODEL_PATH=/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-8B-to-mcore

bash /workspace/recipe/run_qwen3_8b_pt.sh 2>&1 | tee /workspace/recipe/run_qwen3_8b_pt_mutil_node_rank_0.log
```

**Rank 1 节点：**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=2                     # 总节点数
export RANK=1                           # 当前节点 rank
export MASTER_ADDR=<MASTER_NODE_IP>     # 主节点 IP（与 Rank 0 相同）
export MASTER_PORT=23567                # 通信端口（与 Rank 0 相同）
export KUBERNETES_CONTAINER_RESOURCE_GPU=8
export HF_MODEL_PATH=/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-8B-to-mcore

bash /workspace/recipe/run_qwen3_8b_pt.sh 2>&1 | tee /workspace/recipe/run_qwen3_8b_pt_mutil_node_rank_1.log
```

**注意：** 多机训练时，需要在所有节点上同时启动训练脚本，确保 `MASTER_ADDR` 和 `MASTER_PORT` 一致。

#### 训练日志

- **单机运行日志：** [run_qwen3_8b_pt_single_node.log](./run_qwen3_8b_pt_single_node.log)
- **多机运行日志（Rank 0）：** [run_qwen3_8b_pt_mutil_node_rank_0.log](./run_qwen3_8b_pt_mutil_node_rank_0.log)
- **多机运行日志（Rank 1）：** [run_qwen3_8b_pt_mutil_node_rank_1.log](./run_qwen3_8b_pt_mutil_node_rank_1.log)

### Qwen3-VL-30B-A3B-Instruct 训练

对 Qwen3-VL-30B-A3B-Instruct 模型进行训练。

#### 环境准备

```bash
docker exec -it danerli_benchmark bash
```

#### 单机训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23567
export GPUS_PER_NODE=8
export PRETRAIN_CHECKPOINT_PATH=/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-VL-30B-A3B-Instruct-to-mcore

bash /workspace/recipe/run_qwen3vl_30B_A3B_pt.sh 2>&1 | tee /workspace/recipe/Qwen3-VL-30B-A3B-Instruct_single_node.log
```

#### 多机训练

**Rank 0 节点：**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES=2
export NODE_RANK=0
export MASTER_ADDR=${MASTER_ADDR:-"请设置 MASTER_ADDR 环境变量"}  # 主节点 IP（必需） 
export MASTER_PORT=${MASTER_PORT:-23567}                        # 通信端口                      
export GPUS_PER_NODE=8 
export TP=2
export PP=8
export GBS=2
export PRETRAIN_CHECKPOINT_PATH=/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-VL-30B-A3B-Instruct-to-mcore
bash /workspace/recipe/run_qwen3vl_30B_A3B_pt_mutil_node_rank_0.sh 2>&1 | tee /workspace/recipe/Qwen3-VL-30B-A3B-Instruct_mutil_node_rank_0.log
```

**Rank 1 节点：**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES=2
export NODE_RANK=1
export MASTER_ADDR=${MASTER_ADDR:-"请设置 MASTER_ADDR 环境变量"}  # 主节点 IP（必需） 
export MASTER_PORT=${MASTER_PORT:-23567}                        # 通信端口  
export GPUS_PER_NODE=8 
export TP=2
export PP=8
export GBS=2
export PRETRAIN_CHECKPOINT_PATH=/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-VL-30B-A3B-Instruct-to-mcore
bash /workspace/recipe/run_qwen3vl_30B_A3B_pt_mutil_node_rank_1.sh 2>&1 | tee /workspace/recipe/Qwen3-VL-30B-A3B-Instruct_mutil_node_rank_1.log
```

**注意：** 多机训练脚本中已包含必要的环境变量配置，只需设置 `PRETRAIN_CHECKPOINT_PATH` 即可。

#### 训练日志

- **单机运行日志：** [Qwen3-VL-30B-A3B-Instruct_single_node.log](./Qwen3-VL-30B-A3B-Instruct_single_node.log)
- **多机运行日志（Rank 0）：** [Qwen3-VL-30B-A3B-Instruct_mutil_node_rank_0.log](./Qwen3-VL-30B-A3B-Instruct_mutil_node_rank_0.log)
- **多机运行日志（Rank 1）：** [Qwen3-VL-30B-A3B-Instruct_mutil_node_rank_1.log](./Qwen3-VL-30B-A3B-Instruct_mutil_node_rank_1.log)

---

## ❓ 常见问题

### 1. Docker 容器无法访问 GPU

**问题：** 运行 `nvidia-smi` 在容器内无法看到 GPU。

**解决方案：**
- 确保安装了 `nvidia-docker2`：`sudo apt-get install -y nvidia-docker2`
- 重启 Docker 服务：`sudo systemctl restart docker`
- 检查 Docker 运行时：`docker info | grep -i runtime`

### 2. 多机训练时 Ray 连接失败

**问题：** 工作节点无法连接到主节点的 Ray 服务。

**解决方案：**
- 检查防火墙设置，确保端口 `23467` 和 `20122-20999` 开放
- 验证网络连通性：`ping <MASTER_ADDR>`
- 确认 `MASTER_ADDR` 和 `MASTER_PORT` 在所有节点上一致
- 检查 `NCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME` 配置是否正确

### 3. 训练过程中出现 OOM（内存不足）

**问题：** GPU 内存不足导致训练失败。

**解决方案：**
- 减小 `batch_size` 或 `micro_batch_size`
- 使用梯度累积（gradient accumulation）
- 启用 CPU offload（如果支持）
- 检查是否有其他进程占用 GPU 内存

### 4. Flash Attention 安装失败

**问题：** `pip install flash_attn` 编译失败。

**解决方案：**
- 确保 CUDA 版本兼容（推荐 CUDA 11.8+）
- 使用 `--no-build-isolation` 参数：`pip install flash_attn==2.5.0 --no-build-isolation`
- 如果仍然失败，可以尝试从源码编译

### 5. 模型路径不存在

**问题：** 训练时提示模型路径不存在。

**解决方案：**
- 确认预训练模型已下载到指定路径
- 检查路径权限：`ls -l <MODEL_PATH>`
- 验证模型格式是否为 HuggingFace 格式

### 6. 评测脚本无法启动 vLLM 服务

**问题：** 模型评测时 vLLM 服务启动失败。

**解决方案：**
- 检查端口是否被占用：`lsof -i :8000`
- 确认 GPU 数量足够（大模型可能需要多卡）
- 检查模型路径和格式是否正确
- 查看日志文件排查具体错误

### 7. 数据下载失败

**问题：** 下载训练数据时网络超时或失败。

**解决方案：**
- 检查网络连接
- 尝试使用代理或镜像源
- 手动下载后放置到指定目录
- 验证下载链接是否仍然有效

---

## 📝 注意事项

1. **路径配置：** 所有路径配置请根据实际环境修改，特别是挂载路径和模型路径
2. **网络配置：** 多机训练时，确保所有节点在同一网络内，且端口未被占用
3. **资源要求：** 大模型训练需要大量 GPU 内存和存储空间，请提前规划资源
4. **日志保存：** 训练日志会自动保存，建议定期检查日志以监控训练进度
5. **Checkpoint 管理：** 训练过程中会生成多个 checkpoint，注意磁盘空间管理
6. **环境隔离：** 建议为不同项目使用不同的 Docker 容器，避免环境冲突

---

## 📚 相关资源

- [Qwen 官方文档](https://github.com/QwenLM/Qwen)
- [Megatron-LM 文档](https://github.com/NVIDIA/Megatron-LM)
- [Ray 分布式训练文档](https://docs.ray.io/)

---

## 📧 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。
