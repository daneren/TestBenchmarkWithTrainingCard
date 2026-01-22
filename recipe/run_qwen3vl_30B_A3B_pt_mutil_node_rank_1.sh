export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_GID_INDEX=3

export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_NET_SHARED_BUFFERS=0
export NCCL_NET_GDR_LEVEL=2

bash /workspace/recipe/run_qwen3vl_30B_A3B_pt.sh 2>&1 | tee Qwen3-VL-30B-A3B-Instruct_mutil_node_rank_1.log


