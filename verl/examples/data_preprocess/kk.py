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

INSTRUCT_PROMPT = """You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>."""

def generate_base_prompt(dp, template_type):
    quiz = dp['quiz']
    if template_type == 'base':
        # After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. 
        
        prefix = f"""The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. User:The final conclusion can only be determined after thinking and verifying its correctness. The identity of each character should be clearly written in the <answer> </answer> tags, and other thinking and verifying content should be in the <think> </think> tags. List the identity of each person one by one, for example, <answer> (1) Zoey is a knave\n(2) Oliver is a knight\n(3)... </answer>.\n{quiz}\nAssistant:<think>"""

        # prefix = f"""The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind to get a possible answer, then carefully verifies the answer according to the question, and finally provides the user with the final answer. The reasoning process, verification process and final answer are enclosed within the <think> </think>, <verify> </verify> and <answer> </answer> tags, respectively, i.e., <think>reasoning process</think><verify>verification process</verify><answer>final answer</answer>. Now the user asks you to solve a logical reasoning problem. The assistant need to think about the reasoning and verify the results according to the original question. If it is correct, output the final conclusion. The identity of each character should be clearly written in the <answer> </answer> tag. List the identity of each person one by one, for example, <answer> (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>.\n\nUser:{quiz}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        raise ValueError("We should use apply_chat_template to apply chat template")
        # prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', type=str, required=True)
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--train_size', type=int, default=900)
    parser.add_argument('--test_size', type=int, default=100)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    
    args = parser.parse_args()
    
    data_source = 'kk_logic'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Load custom JSONL dataset
    def gen_from_jsonl(path):
        with open(path) as f:
            for line in f:
                yield json.loads(line)
    
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
    print(len(raw_dataset))

    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    apply_chat_template = True
    if args.template_type == "base":
        apply_chat_template = False
        
    def make_map_fn(split, apply_chat_template):
        def process_fn(example, idx):
            if not apply_chat_template:
                prompt = generate_base_prompt(example, template_type=args.template_type)
            else:
                prompt = [
                    {
                        "role": "system",
                        "content": INSTRUCT_PROMPT
                    },
                    {
                        "role": "user",
                        "content": example["quiz"],
                    }
                ]
                
            solution = {
                "solution_text_format": example['solution_text_format'],
                "statements": example['statements']
            }
            data = {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "logic",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
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
    test_dataset = test_dataset.map(function=make_map_fn('test', apply_chat_template=apply_chat_template), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    print(train_dataset[0])

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)