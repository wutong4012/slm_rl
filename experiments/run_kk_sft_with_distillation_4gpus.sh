set -x

nproc_per_node=4
ROOT=../reft/
project_name=logic_rl
experiment_name=distill_into_small
save_path=$ROOT/experiments/checkpoints/$project_name/$experiment_name

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$ROOT/experiments/dataset_distilled_from_7b/kk/instruct/5ppl/sft.parquet \
    data.val_files=$ROOT/experiments/dataset_distilled_from_7b/kk/instruct/5ppl/sft_val.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    optim.lr=1e-5 \
    optim.warmup_steps_ratio=0.0 \
    data.max_length=2600 \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=16 \
    model.partial_pretrain=$ROOT/../models/Qwen2.5-1.5B-Instruct \
    model.fsdp_config.cpu_offload=true \
    model.fsdp_config.offload_params=true \
    model.enable_gradient_checkpointing=true \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=4 \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true $@ 2>&1 | tee $experiment_name.log
