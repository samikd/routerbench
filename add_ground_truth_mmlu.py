import json
from datasets import load_dataset
from tqdm.auto import tqdm

def append_ground_truth(jsonl_file: str, mmlu_data_file: str, output_jsonl_file: str):
    """Extracts last question from JSONL, finds ground truth, and appends it."""
    with open(mmlu_data_file, "r", encoding="utf-8") as f:
        mmlu_data = json.load(f)
    
    updated_data = []
    unknown_count = 0
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            if "messages" in sample and sample["messages"]:
                user_content = sample["messages"][0]["content"]  # Extract user content
                questions = user_content.split("\n\nPlease answer with the letter of the correct answer.\n\n")  # Split into questions
                last_question = questions[-1].split("\nA)")[0].strip()   # Get last question without choices .split("\nA)")[0].strip() 

                ground_truth = mmlu_data.get(last_question, "Unknown")
                
                if ground_truth=='Unknown':
                    unknown_count +=1

                sample["messages"].append({
                    "role": "ground_truth",
                    "content": ground_truth
                })
            
            updated_data.append(sample)
    
    print('unknown count', unknown_count)
    with open(output_jsonl_file, "w", encoding="utf-8") as f:
        for sample in updated_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"Updated JSONL saved to {output_jsonl_file}")

# Define files
mmlu_output_file = "all_mmlu_splits.json"
jsonl_input_file = "mmlu_gpt-4-1106-preview_test.jsonl"
jsonl_output_file = "mmlu_gpt-4-1106-preview_test_output.jsonl"


# Append ground truth to JSONL file
append_ground_truth(jsonl_input_file, mmlu_output_file, jsonl_output_file)

