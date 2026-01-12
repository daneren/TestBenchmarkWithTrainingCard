from dotenv import dotenv_values
env = dotenv_values('.env')
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType


# 添加model 和 api_url传入代码
import argparse

parser = argparse.ArgumentParser(description="Set model and api_url for evaluation task.")
parser.add_argument('--model', type=str, default=env.get("MODEL", "DAPO-Qwen3-8B-VLLM-BF16-ROLLOUT-bsz16-p_no_shuffle-r_20k-step50"), help="Model name")
parser.add_argument('--api_url', type=str, default=env.get("API_URL", "http://0.0.0.0:8000/v1/chat/completions"), help="API URL")

args = parser.parse_args()
model = args.model
api_url = args.api_url


task_cfg = TaskConfig(
    model=model,
    api_url=api_url,

    api_key="1",
    eval_type=EvalType.SERVICE,
    datasets=[
        # 'mmlu_pro',
        # 'gsm8k',
        # 'mmlu',
        'mmmu_pro',
    ],
    dataset_args={
        'mmlu_pro': {
            'subset_list': ['math']  # 只评测 math 子集
        }
    },
    eval_batch_size=5,
    generation_config={
        'max_tokens': 9000,  # Maximum number of tokens to generate, recommended to set to a large value to avoid output truncation
        'temperature': 0.6,  # Sampling temperature (recommended value from Qwen report)
        'top_p': 0.95,  # Top-p sampling (recommended value from Qwen report)
        'top_k': 20,  # Top-k sampling (recommended value from Qwen report)
        'n': 1,  # Number of responses generated per request
    },
    rerun_review=True,
    work_dir='/workspace/recipe/model_evaluation_result/',
    analysis_report=True,
    # limit=100,  # Set to 100 data points for testing
)

run_task(task_cfg=task_cfg)