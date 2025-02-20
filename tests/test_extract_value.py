import re
import pandas as pd
import os
import glob

def extract_predicted_output(predict_str: str) -> str:
    """
    Extracts the predicted value from the input string based on the following rules:

    format: <think>...</think>......output here ......
    """
    
    think_match = re.search(r'<think>(.*?)</think>', predict_str, flags=re.DOTALL)
    
    if not think_match:
        return predict_str
    
    end_index = think_match.end()
    if end_index == len(predict_str):
        return predict_str
    
    output_content = predict_str[end_index:]
    return output_content
def extract_nan_predictions(df, output_dir='nan_outputs'):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 找出prediction为nan的行
    nan_rows = df[df['prediction'].isna()]
    
    # 为每个nan的输出创建单独的文件
    for idx, row in nan_rows.iterrows():
        # 使用索引作为文件名，确保唯一性
        filename = f'nan_output_{idx}.txt'
        filepath = os.path.join(output_dir, filename)
        
        # 将model_output写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(row['model_output']))
    
    print(f"已提取 {len(nan_rows)} 个nan预测的输出到 {output_dir} 目录")
    return nan_rows
def extract_predicted_value(predict_str: str) -> str:
    """
    Extracts the predicted value from the input string based on the following rules:

    format: <think>...</think>XXXXX/boxed{answer here}XXXXX

    1) Look for <think> and </think>
        if not found:
            return empty string
    2) Look for /boxed{}
           a) \boxed{\text{...}...} and extract '...' plus any additional text.
           b) \text{\boxed{...}} and extract '\boxed{...}'.
           c) \boxed{...} and extract '...'.
       - If no \boxed{...}:
        return empty string
    
    Examples:
      - "\\boxed{42}" -> "42"
      - "\\boxed{\\text{Light blue}}" -> "Light blue"
      - "<think>xxx</think>\\boxed{42}" -> "42"
      - "<think>}</think>\\boxed{\\text{Light blue}" -> "Light blue"
      - "no boxed content" -> ""
    """
    
    # Step 1: Search for the <think>...</think> block
    think_match = re.search(r'<think>(.*?)</think>', predict_str, flags=re.DOTALL)
    
    if not think_match:
        return ""
    
    # Step 2: Get the end index of the </think> tag
    end_index = think_match.end()

    # # Step 3: Extract the content after the </think> tag
    output_content = predict_str[end_index:]

    # Step 4: Search for \boxed{\text{...}...} with additional text
    box_text_extra_match = re.search(r'\\boxed\{\\text\{(.*?)\}(.*?)\}',output_content, flags=re.DOTALL)
    if box_text_extra_match:
        text_part = box_text_extra_match.group(1)
        text_part = text_part.strip() if text_part is not None else ""
        extra_part = box_text_extra_match.group(2)
        extra_part = extra_part.strip() if extra_part is not None else ""
        # Concatenate with a space if both parts are present
        if extra_part:
            return f"{text_part} {extra_part}"
        return text_part
    
    # Step 2b: Search for \boxed{\text{...}} without additional text
    box_text_match = re.search(r'\\boxed\{\\text\{(.*?)\}\}', output_content, flags=re.DOTALL)
    if box_text_match:
        matched_text = box_text_match.group(1)
        return matched_text.strip() if matched_text is not None else ""
    
    # Step 2c: Search for \text{\boxed{...}} and reconstruct \boxed{...}
    text_box_match = re.search(r'\\text\{\\boxed\{(.*?)\}\}', output_content, flags=re.DOTALL)
    if text_box_match:
        matched_text = text_box_match.group(1)
        matched_text = matched_text.strip() if matched_text is not None else ""
        # Reconstruct \boxed{...}
        return f'\\boxed{{{matched_text}}}'
    
    # Step 2d: Search for \boxed{...} without \text{...}
    box_match = re.search(r'\\boxed\{(.*?)\}', output_content, flags=re.DOTALL)
    if box_match:
        matched_text = box_match.group(1)
        return matched_text.strip() if matched_text is not None else ""

    # If no \boxed{...} found, return empty string
    return ""
if __name__ == "__main__":
    
    # 1. 抽出结果文件里面nan的原始模型输出
    folder = "/opt/tiger/VLMEvalKit/nan_outputs"
    # df = pd.read_excel('/opt/tiger/VLMEvalKit/tests/Qwen2.5-VL-Instruct_warmup_v2_MMMU_DEV_VAL_openai_result.xlsx')
    # nan_records = extract_nan_predictions(df, output_dir=folder)

    to_check_folder = "/opt/tiger/VLMEvalKit/cases"
    # 2. 统计为nan的情况数量
    cases = glob.glob(folder+"/*.txt")
    cnts = {
        "api_fail": 0,
        "extract_fail": 0,
        "extract_fail but box exists": 0
    }
    for case in cases:
        with open(case, "r") as file:
            predict_str = file.read()
        if predict_str.strip() == "Failed to obtain answer via API.":
            cnts["api_fail"] += 1
            continue
        value = extract_predicted_value(predict_str)
        output = extract_predicted_output(predict_str)
        if not value or len(value)==0:
            cnts["extract_fail"] += 1
            # 3. 收集一下有boxed但是没有抽出的来的情况
            if "\\boxed" in predict_str:
                # save to to_check_folder
                with open(os.path.join(to_check_folder, os.path.basename(case)), "w") as file:
                    file.write(predict_str)
                cnts["extract_fail but box exists"] += 1

    print(f"Total nan files:{len(cases)}")
    print(f"Total api fail:{cnts['api_fail']}")
    print(f"Total extract fail:{cnts['extract_fail']}")
    print(f"Total extract fail but box exists:{cnts['extract_fail but box exists']}")