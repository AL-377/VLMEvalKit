import pandas as pd
import os

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

# 读取Excel文件
df = pd.read_excel('/opt/tiger/VLMEvalKit/tests/Qwen2.5-VL-Instruct_warmup_v2_MMMU_DEV_VAL_openai_result.xlsx')

# 调用函数
nan_records = extract_nan_predictions(df, output_dir='nan_outputs')

# 打印统计信息
print(f"总共有 {len(nan_records)} 条nan预测")
print(nan_records)
