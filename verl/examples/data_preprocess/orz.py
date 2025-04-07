# ==========================================================
# Based on code from [Logic-RL] - [https://github.com/Unakar/Logic-RL]
# Modifications: Use `apply_chat_template` instead of hardcoding.
# ==========================================================

""" Preprocess dataset for knights and knaves logic task """
import os
os.environ["HF_HOME"] = "/data/wutong/tmp/huggingface_cache"
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', type=str, required=True)
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    
    args = parser.parse_args()
    
    def gen_from_json():
        dialogues = []
        with open("/home/wutong/data/reft-exp/experiments/raw/orz_math_57k_collected.json", "r") as f:
            dialogues.extend(json.load(f))
        with open("/home/wutong/data/reft-exp/experiments/raw/orz_math_72k_collection_extended.json", "r") as f:
            dialogues.extend(json.load(f))
        
        for item in dialogues[:500]:
            data = {
                "prompt": item[0]["value"],
                "answer": item[1]["ground_truth"]["value"]
            }
            yield data
    
    train_dataset = Dataset.from_generator(gen_from_json)
    print(len(train_dataset))

    def gen_from_json_math500():
        dialogues = []
        with open("/home/wutong/data/reft-exp/experiments/raw/gpqa_diamond.json", "r") as f:
            dialogues.extend(json.load(f))
        
        for item in dialogues:
            yield item

    test_dataset = Dataset.from_generator(gen_from_json_math500)
    print(len(test_dataset))

    apply_chat_template = True
    if args.template_type == "base":
        apply_chat_template = False
        
    def make_map_fn(split, apply_chat_template):
        def process_fn(example, idx):
            if not apply_chat_template:
                prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag. {example["prompt"]}\nAssistant: <think>"""

            data = {
                "data_source": 'orz',
                "prompt": prompt,
                "ability": "logic",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example["answer"]
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'apply_chat_template': apply_chat_template
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train', apply_chat_template=apply_chat_template), with_indices=True)

    def make_map_fn_test(split, apply_chat_template):
        def process_fn(example, idx):
            if not apply_chat_template:
                prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag. {example["prompt"]}\nAssistant: <think>"""

            data = {
                "data_source": 'orz',
                "prompt": prompt,
                "ability": "logic",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example["final_answer"]
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'apply_chat_template': apply_chat_template
                }
            }
            return data
        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn_test('test', apply_chat_template=apply_chat_template), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    print(train_dataset[0])
    print(test_dataset[0])

    train_dataset.to_parquet(os.path.join(local_dir, 'train_debug.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test_gpqa_diamond.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)