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

def language_mix_reward(solution_str: str) -> float:
    """
    Calculate the language mix reward value based on the text content

    Parameters:
        text: Input string

    Return:
        float: Reward value, negative value indicates penalty
    """
    ranges_to_check = [
        (0x4E00, 0x9FFF),   # CJK统一汉字
        (0x3040, 0x30FF),   # 日文平假名和片假名
        (0x3400, 0x4DBF),   # CJK扩展A
        (0x20000, 0x2A6DF), # CJK扩展B
        (0x2A700, 0x2B73F), # CJK扩展C
        (0x2B740, 0x2B81F), # CJK扩展D
        (0x2B820, 0x2CEAF), # CJK扩展E
        (0xF900, 0xFAFF),   # CJK兼容汉字
        (0xAC00, 0xD7AF),   # 韩文音节
        (0x1100, 0x11FF),   # 韩文字母
        (0x3130, 0x318F),   # 韩文兼容字母
        (0x0400, 0x04FF),   # 西里尔字母
        (0x0500, 0x052F),   # 西里尔字母补充
        (0x2DE0, 0x2DFF),   # 西里尔字母扩展-A
        (0xA640, 0xA69F),   # 西里尔字母扩展-B
        (0x0600, 0x06FF),   # 阿拉伯文
        (0x0750, 0x077F),   # 阿拉伯文补充
        (0x0900, 0x097F),   # 天城文
        (0x0A80, 0x0AFF),   # 古吉拉特文
        (0x0B00, 0x0B7F),   # 奥里亚文
        (0x0980, 0x09FF),   # 孟加拉文
        (0x0A00, 0x0A7F),   # 锡克教文
        (0x0C00, 0x0C7F),   # 泰卢固文
        (0x0C80, 0x0CFF),   # 卡纳达文
        (0x0D00, 0x0D7F),   # 马拉雅拉姆文
        (0x0E00, 0x0E7F),   # 泰文
        (0x0E80, 0x0EFF),   # 老挝文
        (0x0F00, 0x0FFF),   # 藏文
        (0x1000, 0x109F),   # 缅甸文
        (0x1200, 0x137F),   # 埃塞俄比亚文
        (0x1780, 0x17FF),   # 高棉文
        (0x1800, 0x18AF),   # 蒙古文
    ]
    
    for char in solution_str:
        char_code = ord(char)
        
        for start, end in ranges_to_check:
            if start <= char_code <= end:
                return -5.0
    
    return 0.0

def logic_reason_reward(processed_str: str) -> float:
    """
    Calculate the logic reason reward value based on the text content
    """

    logical_operators = [
        "∨", "\\u2228",  # OR
        "∧", "\\u2227",  # AND
        "¬", "\\u00AC",  # NOT
        # "→", "\\u2192",  # IMPLIES
        # "↔", "\\u2194",  # IFF (等价)
        # "&", "\\u0026",  # AND (ASCII)
        # "|", "\\u007C",  # OR (ASCII)
        # "⊕", "\\u2295",  # XOR
        # "⊥", "\\u22A5",  # BOTTOM (矛盾)
        # "⊤", "\\u22A4",  # TOP (重言式)
    ]

    for op in logical_operators:
        if op in processed_str:
            print(op)
            return 0.5

    return -0.1

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'verify_start': ('<verify>', 1),
        'verify_end': ('</verify>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['verify_start'] - (positions['think_end'] + len('</think>')) > 1 or
        # positions['think_end'] > positions['verify_start'] or
        positions['verify_start'] > positions['verify_end'] or
        positions['answer_start'] - (positions['verify_end'] + len('</verify>')) > 1 or
        # positions['verify_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><verify>...</verify><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed

def get_thinking_reward(processed_str: str, expected_names: List) -> float:
    """
    Calculate the length and reflection rewards for the thinking part of the response
    """
    think_start = processed_str.find('<think>')
    think_end = processed_str.find('</verify>')

    think_content = processed_str[think_start + len('<think>'):think_end]

    thinking_factor = 1.0
    
    for item in expected_names:
        if item not in think_content:
            thinking_factor *= 0.5

    if thinking_factor == 1.0:
        reflection_tokens = [
            "wait", "yet", "re-evaluate", "review",
            "∨", "\\u2228",  # OR
            "∧", "\\u2227",  # AND
            "¬", "\\u00AC",  # NOT
        ]
        text_lower = think_content.lower()
        
        for token in reflection_tokens:
            if token in text_lower:
                thinking_factor += 1e-5

    return thinking_factor

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

    # Validate language mix
    language_mix_score = language_mix_reward(solution_str)
    if language_mix_score < 0:
        print("\n[Language Mix Penalty]")
        print("  Detected non-English characters in the response")
        print(f"  Language Mix Score: {language_mix_score}")

    # Validate logical reasoning
    # logic_reason_score = logic_reason_reward(solution_str)
    # if logic_reason_score > 0:
    #     print("\n[Logic Reason Reward]")
    #     print("  Detected logical operators in the response")
    #     print(f"  Logic Reason Score: {logic_reason_score}")

    # Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = 0.0 if format_correct else -1.0
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    # Validate answer content
    answer_score = 0
    thinking_factor = 1.0
    if format_correct and answer_text and valid_response_length > 500:
        pred_status = parse_model_answer(answer_text, expected_names)
        if pred_status:
            print(f"\n[Content Validation]")
            print(f"  Expected: {gt_status}")
            print(f"  Predicted: {pred_status}")
            if config.writer.enable:
                write_content['Expected'] = gt_status
                write_content['Predicted'] = pred_status
            
            if pred_status == gt_status:
                thinking_factor = get_thinking_reward(processed_str, expected_names)
                answer_score = 2 * thinking_factor
                # answer_score = 2
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

    # simple overlong and overshort penalty
    length_score = 0
    if valid_response_length <= 500 or valid_response_length >= config.data.max_response_length:
        length_score = -1.0
        print("\n[Overlong and Overshort Penalty]")
    # else:
    #     length_score = ((int(valid_response_length) - 500) / (2048 - 500)) ** (math.log(0.8) / math.log((1024 - 500) / (2048 - 500)))

    if config.writer.enable:
        write_content['Format Score'] = format_score
        write_content['Thinking Factor'] = thinking_factor
        write_content['Language Mix Score'] = language_mix_score
        # write_content['Logic Reason Score'] = logic_reason_score
        write_content['Length Score'] = length_score
        write_content['Answer Score'] = answer_score

    total_score = format_score + answer_score + language_mix_score + length_score  # + logic_reason_score

    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Thinking: {thinking_factor}")
    print(f"  Language Mix: {language_mix_score}")
    # print(f"  Logic Reason: {logic_reason_score}")
    print(f"  Length: {length_score}")
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
            "language_mix_score": language_mix_score,
            "format_score": format_score,
            "thinking_factor": thinking_factor,
            # "logic_reason_score": logic_reason_score,
            "length_score": length_score
        }
    }

    return output
