import re
import os
import math
import json
from typing import Dict, Tuple, Optional, List

from omegaconf import OmegaConf

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
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary.
    
    Args:
        solution_text: Formatted solution text from dataset
        
    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    print("\n[Ground Truth Parsing]")
    
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Warning] Unparseable line: '{line}'")
    
    return status_dict

def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.
    
    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification
        
    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    print("\n[Model Answer Parsing]")
    print(f"  Expected characters: {expected_names}")

    knight_count = answer_text.lower().count('knight')
    knave_count = answer_text.lower().count('knave')

    print(f"  Number of predicted roles: {knight_count + knave_count}")
    if knight_count + knave_count != len(expected_names):
        print(f"  [Error] Number of characters mismatch: {knight_count + knave_count} != {len(expected_names)}")
        return None

    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
            re.IGNORECASE
        )
        match = pattern.search(answer_text)
        
        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Error] Missing identification for {name}")
            return None
    
    return status_dict

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

    # Parse ground truth data
    solution_text = ground_truth.get('solution_text_format', '')
    gt_status = parse_solution_text_format(solution_text)
    expected_names = list(gt_status.keys())
    print(f"[Ground Truth] Final identities: {gt_status}")

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")
    if config.writer.enable:
        write_content['Model Response'] = processed_str

    # Validate answer content
    answer_score = 0
    answer_score_scaled = 0
    if answer_text:
        pred_status = parse_model_answer(answer_text, expected_names)
        if pred_status:
            print(f"\n[Content Validation]")
            print(f"  Expected: {gt_status}")
            print(f"  Predicted: {pred_status}")
            if config.writer.enable:
                write_content['Expected'] = gt_status
                write_content['Predicted'] = pred_status
            
            if pred_status == gt_status:
                answer_score = 1
                answer_score_scaled = 1
                print("  Content validation: FULL MATCH")
    #         else:
    #             answer_score = -0.5
    #             answer_score_scaled = 0
    #             print("  Content validation: MISMATCH")
    #     else:
    #         answer_score = -1
    #         answer_score_scaled = 0
    #         print( "Fail to parse answer")
    # else:
    #     answer_score = -2
    #     answer_score_scaled = 0
    #     print("\n[Content Validation] Skipped due to format errors or missing answer")

    if config.writer.enable:
        write_content['Answer Score'] = answer_score

    total_score = answer_score

    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Answer: {answer_score}")
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
        }
    }

    return output
