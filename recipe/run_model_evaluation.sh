#!/bin/bash

set -xeuo pipefail

# ============================================
# 模型评测一键执行脚本
# 使用方法: ./run_model_evaluation.sh [GPU_ID] [MODEL_PATH] [MODEL_NAME]
# ============================================

# 设置默认值
GPU_ID=${GPU_ID:-"0,1,2,3,4,5,6,7"}
MODEL_PATH=${MODEL_PATH:-"/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-VL-8B-Instruct"}
MODEL_NAME=${MODEL_NAME:-"qwen3_vl_8b_orig"}
PORT=${PORT:-8000}

# 工作目录
WORKSPACE="/workspace"

echo "============================================"
echo "模型评测配置"
echo "============================================"
echo "GPU ID: $GPU_ID"
echo "模型路径: $MODEL_PATH"
echo "模型名称: $MODEL_NAME"
echo "vllm port: $PORT"
echo "============================================"

# 设置GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 设置环境变量
export VERL_LOGGING_LEVEL=DEBUG
export VLLM_CONFIGURE_LOGGING=1
export VLLM_USE_V1=1
export VLLM_USE_DEEP_GEMM=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export WANDB_MODE=offline
export PYTORCH_SYMMETRIC_MEMORY=0


gpu_count=$(echo ${CUDA_VISIBLE_DEVICES:-} | tr ',' '\n' | grep -c .) && gpu_count=$(( gpu_count == 0 ? 8 : gpu_count )) && echo $gpu_count

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

mkdir -vp ./tmp

echo ""
echo "启动 vLLM 服务..."
echo ""

# 启动 vLLM 服务（后台运行）
# 启动文本模型
# VLLM_LOGGING_LEVEL=DEBUG vllm serve "$MODEL_PATH" \
#     --served-model-name "$MODEL_NAME" \
#     --dtype bfloat16 \
#     --load_format safetensors \
#     --max_model_len 21504 \
#     --max_num_seqs 256 \
#     --enable_chunked_prefill \
#     --max_num_batched_tokens 32768 \
#     --enable_prefix_caching \
#     --enable_sleep_mode \
#     --disable_custom_all_reduce \
#     --gpu_memory_utilization 0.8 \
#     --disable_log_stats \
#     --tensor_parallel_size $gpu_count \
#     --seed 0 \
#     --port=$PORT \
#     --override_generation_config '{"temperature": 1.0, "top_k": -1, "top_p": 1.0, "repetition_penalty": 1.0, "max_new_tokens": 20480}' \
#     > ./tmp/vllm_serve_${MODEL_NAME}.log 2>&1 &

# 启动多模态模型
VLLM_LOGGING_LEVEL=DEBUG vllm serve \
    $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size $gpu_count  \
    --mm-encoder-tp-mode data \
    --async-scheduling        \
    --max-model-len 262144    \
    --port=$PORT              \
    --gpu-memory-utilization 0.6 \
    > ./tmp/vllm_serve_${MODEL_NAME}.log 2>&1 &


VLLM_PID=$!
echo "vLLM 服务已启动，PID: $VLLM_PID"
echo "日志文件: ./tmp/vllm_serve_${MODEL_NAME}.log"

# 等待服务启动
echo ""
echo "等待 vLLM 服务启动..."
sleep 300

# 检查服务是否还在运行
if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "错误: vLLM 服务启动失败，请查看日志: ./tmp/vllm_serve_${MODEL_NAME}.log"
    exit 1
fi

echo "vLLM 服务运行正常"
echo ""

# 运行评估脚本
echo "开始运行模型评估..."
cd "$WORKSPACE/recipe"
python3 "$WORKSPACE/recipe/model_evaluate.py" --model "$MODEL_NAME" --api_url http://0.0.0.0:$PORT/v1/chat/completions

EVAL_EXIT_CODE=$?

# 清理：停止 vLLM 服务
echo ""
echo "评估完成，正在停止 vLLM 服务..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "vLLM 服务已停止"

# 返回评估脚本的退出码
exit $EVAL_EXIT_CODE

