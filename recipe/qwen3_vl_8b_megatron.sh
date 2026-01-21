set -x

export workspace=/workspace

cd $workspace

ENGINE=${1:-vllm}
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

# dependency: vllm>=0.11.0, megatron-lm>=0.13, mbridge with qwen3vl_cp branch
# environment option1: use a stable container later than docker://verlai/verl:vllm011.dev6 
    # and install mbridge in it by following the instruction in the container
            # pip remove mbridge if you have installed it
            # pip install git+https://github.com/ISEEKYAN/mbridge.git@qwen3vl_cp # for correct mbridge
# environment option2: use container docker://verlai/verl:vllm011.dev_qwenvl_cp
 

export VLLM_ALLREDUCE_USE_SYMM_MEM=0 # for vllm0.11.0 with TP
export WANDB_MODE=offline

export HF_MODEL_PATH=${HF_MODEL_PATH:-"/mnt/cfs/tilearn/pretrain_models/Qwen/Qwen3-VL-8B-Instruct-to-mcore"}

GEN_TP=${GEN_TP:-1}
CP=${CP:-2}
TP=${TP:-2}
PP=${PP:-2}

# nnodes
export nnodes=${nnodes:-1}

# 根据节点数设置实验名称
if [ "$nnodes" -eq 1 ]; then
    export experiment_name="qwen3_vl_8b_megatron_single_node"
    echo "Running in single node mode"
else
    export experiment_name="qwen3_vl_8b_megatron_mutil_node"
    echo "Running in multi-node mode with $nnodes nodes"
fi

echo "Experiment name: $experiment_name"

export exp_path=$workspace/recipe/exp
export CKPTS_DIR=$exp_path/checkpoint/qwen3_vl_8b_megatron_$nnodes_node

train_files=/workspace/datatsets/geo3k/train.parquet
val_files=/workspace/datatsets/geo3k/test.parquet

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_rl_demo' \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$nnodes \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=30 $@ 

