import re
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
    import glob
    cases = glob.glob("cases/*.txt")
    for case in cases:
        with open(case, "r") as file:
            predict_str = file.read()
        print(extract_predicted_value(predict_str))