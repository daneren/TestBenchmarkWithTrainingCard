export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=2 
export RANK=1
export MASTER_ADDR=${MASTER_ADDR:-"请设置 MASTER_ADDR 环境变量"}  # 主节点 IP（必需） 
export MASTER_PORT=${MASTER_PORT:-23567}                        # 通信端口
export KUBERNETES_CONTAINER_RESOURCE_GPU=8 

export workspace=/mnt/cfs/danerli/workspace/test_benchmark/TestBenchmarkWithTrainingCard

bash $workspace/recipe/run_qwen3_8b_pt.sh 2>&1 | tee run_qwen3_8b_pt_mutil_node_rank_1.log