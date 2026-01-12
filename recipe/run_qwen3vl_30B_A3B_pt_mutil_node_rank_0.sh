export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES=2
export NODE_RANK=0
export MASTER_ADDR=${MASTER_ADDR:-"请设置 MASTER_ADDR 环境变量"}  # 主节点 IP（必需）
export MASTER_PORT=${MASTER_PORT:-23567}                        # 通信端口
export GPUS_PER_NODE=8 


export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_GID_INDEX=3

export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_NET_SHARED_BUFFERS=0
export NCCL_NET_GDR_LEVEL=2



bash /mnt/cfs/danerli/workspace/test_benchmark/TestBenchmarkWithTrainingCard/recipe/run_qwen3vl_30B_A3B_pt.sh 2>&1 | tee Qwen3-VL-30B-A3B-Instruct_mutil_node_rank_0.log


