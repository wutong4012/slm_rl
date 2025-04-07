import re
import os
import math
import json
from typing import Dict, Tuple, Optional, List

from omegaconf import OmegaConf
from verl.utils.reward_score.math_utils import is_equal, solution2answer


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        # print("[Error] Failed to locate model response header")
        # return None, solution_str
        processed_str = solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>.*?(\\boxed{.*}).*?</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

    if not matches:
        answer_pattern = r'(\\boxed{.*})'
        matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str

    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def get_thinking_reward(processed_str: str) -> float:
    
    words = re.findall(r'\b\w+\b', processed_str)
    word_count = len(words)
    
    thinking_reward = int(word_count / 100.0) * 0.1
    print(f"  Thinking content word count: {word_count}")

    return thinking_reward

def compute_score(
        solution_str: str, 
        ground_truth: Dict[str, str],
        config,
        valid_response_length: int,
    ) -> Dict[str, float]:
    """Computes comprehensive score for model response.

    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        config: Optional configuration object
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    if config.writer.enable:
        write_content = {}

    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))
    print(f"[Ground Truth] Final identities: {ground_truth}")

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")
    if config.writer.enable:
        write_content['Model Response'] = processed_str

    # Validate response format
    chat_score = 0
    patterns = [r"assistant:", r"user:", r"human:"]
    for pattern in patterns:
        matches = re.findall(pattern, processed_str, re.IGNORECASE)
        if len(matches) > 0:
            print(f"[Error] Invalid response format: {pattern}")
            chat_score = -5
            break

    # Validate answer content
    answer_score = 0
    answer_score_scaled = 0
    if answer_text:
        pred_status = solution2answer(answer_text)
        if pred_status:
            print(f"\n[Content Validation]")
            print(f"  Expected: {solution2answer(ground_truth)}")
            print(f"  Predicted: {pred_status}")
            if config.writer.enable:
                write_content['Expected'] = solution2answer(ground_truth)
                write_content['Predicted'] = pred_status
            
            if is_equal(solution2answer(ground_truth), pred_status):
                answer_score = 1
                answer_score_scaled = 1
                print("  Content validation: FULL MATCH")
            else:
                answer_score = -0.5
                answer_score_scaled = 0
                print("  Content validation: MISMATCH")
        else:
            answer_score = -1
            answer_score_scaled = 0
            print( "Fail to parse answer")
    else:
        answer_score = -2
        answer_score_scaled = 0
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    if config.writer.enable:
        write_content['Answer Score'] = answer_score
        write_content['Chat Score'] = chat_score

    total_score = answer_score + chat_score

    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Answer: {answer_score}")
    print(f"  Chat: {chat_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")

    if config.writer.enable:
        if not os.path.exists(config.writer.log_dir):
            os.makedirs(config.writer.log_dir, exist_ok=True)

        if config.writer.log_config:
            with open(f"{config.writer.log_dir}/config.json", "w") as f:
                json.dump(OmegaConf.to_container(config), f, indent=4)
            config.writer.log_config = False

        with open(f"{config.writer.log_dir}/results_step{config.writer.eval_step}.json", "a") as f:
            json.dump(write_content, f, indent=4)
            f.write("\n")
    
    output = {
        "score": total_score,
        "extra_info": {
            "outcome_score": answer_score_scaled,
            "chat_score": chat_score,
        }
    }

    return output
