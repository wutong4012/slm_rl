# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
import ray
import numpy as np
import hydra
import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    local_path = copy_to_local(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='actor_rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")
    # real_batch_size = data.batch['input_ids'].shape[0]
    config_batch_size = config.data.batch_size
    dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
    print(f"DP size: {dp_size}")
    num_batch = (total_samples // config_batch_size) + 1
    # output_lst = [[] for _ in range(config.data.n_samples)]

    makedirs(config.data.output_dir, exist_ok=True)
    for batch_idx in range(num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        start_index = batch_idx * config_batch_size
        end_index = min((batch_idx + 1) * config_batch_size, total_samples)
        if os.path.exists(os.path.join(config.data.output_path, f"partition_{start_index}_{end_index}.parquet")):
            print(f'Batch {batch_idx}, from {start_index} to {end_index} already exists, skipping.')
            continue
        
        df_sub = dataset.iloc[start_index:end_index, :].copy(deep=True)
        batch_chat_lst = chat_lst[start_index:end_index]
        if not batch_chat_lst:
            break
        # batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(batch_chat_lst,
                                               add_generation_prompt=True,
                                               padding="max_length", # pad to max_length instead of max_length of the batch
                                               truncation=True,
                                               max_length=config.rollout.prompt_length,
                                               return_tensors='pt',
                                               return_dict=True,
                                               tokenize=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        real_batch_size = data.batch['input_ids'].shape[0]
        if real_batch_size % dp_size != 0:
            dummy_data_size = dp_size - real_batch_size % dp_size
            dummy_data = data[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(
                f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
            )

        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
        # START TO GENERATE FOR n_samples TIMES
        # for i in range(config.data.n_samples):
        #     output = wg.generate_sequences(data)
        output = wg.generate_sequences(data)
        rollout_n = config.rollout.n
        max_length = config.rollout.response_length
        assert output.batch.batch_size[0] == batch_size * rollout_n, f'output batch size {output.batch.batch_size[0]} != {batch_size} * {rollout_n}'

        responses = output.batch["responses"].reshape(batch_size, rollout_n, -1)
        print(f"Response shape: {responses.shape}")
        responses_lst = responses.tolist()
        df_sub["raw_responses"] = responses_lst

        input_ids = output.batch["input_ids"].reshape(batch_size, rollout_n, -1)
        print(f"Input IDs shape: {input_ids.shape}")
        input_ids = input_ids.tolist()
        df_sub["input_ids"] = input_ids

        attention_mask = output.batch["attention_mask"].reshape(batch_size, rollout_n, -1)
        print(f"Attention mask shape: {attention_mask.shape}")
        attention_mask = attention_mask.tolist()
        df_sub["attention_mask"] = attention_mask

        position_ids = output.batch["position_ids"].reshape(batch_size, rollout_n, -1)
        print(f"Position IDs shape: {position_ids.shape}")
        position_ids = position_ids.tolist()
        df_sub["position_ids"] = position_ids


        old_log_probs = output.batch["old_log_probs"].reshape(batch_size, rollout_n, -1)
        # print(old_log_probs.shape)
        old_log_probs_lst = old_log_probs.tolist()
        df_sub["old_log_probs"] = old_log_probs_lst
        def reshape_list(input_list, batch_size, rollout_n):
            batched_list = []
            index = 0
            for _ in range(batch_size):
                batch = []
                for _ in range(rollout_n):
                    if index < len(input_list):
                        batch.append(input_list[index])
                        index += 1
                    else:
                        break # Stop if input list is exhausted
                batched_list.append(batch)
            return batched_list

        # output = output[:real_batch_size]
        output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                                 skip_special_tokens=False)

        # remove the padding
        pad_token = tokenizer.pad_token
        output_text_unpad = []
        for text in output_text:
            output_text_unpad.append(text.replace(pad_token, ''))
        output_text_unpad_reshape = reshape_list(output_text_unpad, config_batch_size, rollout_n)
        df_sub["decoded_responses"] = output_text_unpad_reshape
        output_file = os.path.join(config.data.output_dir, f"partition_{start_index}_{end_index}.parquet")
        df_sub.to_parquet(output_file)

        # output_lst[i].extend(output_text_unpad)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    # output_lst = np.array(output_lst, dtype=object)
    # output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    # add to the data frame
    # dataset[f'responses'] = output_lst

    # write to a new parquet
    # output_dir = os.path.dirname(config.data.output_path)
    # makedirs(output_dir, exist_ok=True)
    # dataset.to_parquet(config.data.output_path)

    # return output_text
    # Collect all sub parquet files
    sub_files = [os.path.join(config.data.output_dir, f) for f in os.listdir(config.data.output_dir) if f.endswith('.parquet')]
    # Concatenate all sub parquet files
    combined_df = pd.concat([pd.read_parquet(f) for f in sub_files], ignore_index=True)
    # Write the combined dataframe to a new parquet file
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    combined_df.to_parquet(config.data.output_path)

if __name__ == '__main__':
    main()
