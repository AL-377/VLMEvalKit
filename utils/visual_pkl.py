import pickle
import pandas as pd
import json
from pathlib import Path
import argparse
import numpy as np

def load_pkl(file_path):
    """Load a pickle file and return its contents."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def display_pkl_info(data):
    """Display information about the pickle file contents."""
    print("\n=== Pickle File Contents ===")
    print(f"Data type: {type(data)}")
    
    if isinstance(data, dict):
        print("\nDictionary keys:")
        for key in data.keys():
            print(f"- {key}: {type(data[key])}")
    
    elif isinstance(data, pd.DataFrame):
        print("\nDataFrame info:")
        print(data.info())
        print("\nFirst few rows:")
        print(data.head())
    
    elif isinstance(data, list):
        print(f"\nList length: {len(data)}")
        if len(data) > 0:
            print(f"First element type: {type(data[0])}")
            print("First element preview:", str(data[0])[:200])
    
    else:
        print("\nContent preview:")
        print(str(data)[:500])

def convert_numpy_types(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {(int(k) if isinstance(k, np.integer) else k): convert_numpy_types(v) 
                for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def main():
    parser = argparse.ArgumentParser(description='Visualize pickle file contents')
    parser.add_argument('--file_path', type=str, help='Path to the pickle file')
    parser.add_argument('--save', type=str, help='Save as JSON file (if possible)', default=None)
    args = parser.parse_args()

    try:
        data = load_pkl(args.file_path)
        display_pkl_info(data)
        
        if args.save:
            try:
                # Convert to JSON-serializable format if possible
                if isinstance(data, pd.DataFrame):
                    json_data = data.to_dict(orient='records')
                else:
                    json_data = convert_numpy_types(data)
                    
                with open(args.save, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f"\nSaved to {args.save}")
            except Exception as e:
                print(f"\nFailed to save as JSON: {e}")
                
    except Exception as e:
        print(f"Error loading pickle file: {e}")

if __name__ == "__main__":
    main()
