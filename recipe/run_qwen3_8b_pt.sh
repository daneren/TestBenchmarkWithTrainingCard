export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export KUBERNETES_CONTAINER_RESOURCE_GPU=${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}


export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_GID_INDEX=3

export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_NET_SHARED_BUFFERS=0
export NCCL_NET_GDR_LEVEL=2

export HF_MODEL_PATH=${HF_MODEL_PATH:-"/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-8B-to-mcore"}

export workspace=${workspace:-"/workspace"}
cd $workspace/Pai-Megatron-Patch/examples/qwen3

sh run_mcore_qwen3.sh  \
dlc  \
8B   \
1    \
8 \
1e-5   \
1e-6   \
4096  \
4096  \
bf16  \
4   \
2  \
1 \
1 \
1 \
true \
true   \
true \
false \
sel   \
false \
100000  \
${workspace}/datatsets/qwen-datasets/mmap_qwen3_datasets_text_document   \
${workspace}/datatsets/qwen-datasets/mmap_qwen3_datasets_text_document   \
$HF_MODEL_PATH  \
100000000  \
1000000   \
${workspace}/test_logs/output_mcore_qwen3_8b_pt