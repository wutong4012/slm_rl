set -x

CHECKPOINT_PATH=checkpoints/logic_rl/grpo_run/global_step_380/actor/huggingface
DATA_PATH=dataset/kk/instruct/5ppl/train.parquet
OUTPUT_PATH=dataset_distilled_from_7b/kk/instruct/5ppl/sft.parquet
OUTPUT_DIR=dataset_distilled_from_7b/kk/instruct/5ppl/temp_val

export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    data.path=$DATA_PATH \
    data.prompt_key=prompt \
    data.output_path=$OUTPUT_PATH \
    data.output_dir=$OUTPUT_DIR \
    model.path=$CHECKPOINT_PATH \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=512 \
    rollout.response_length=2048 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.7 \
    rollout.max_model_len=2600 \
    rollout.n=5 \
