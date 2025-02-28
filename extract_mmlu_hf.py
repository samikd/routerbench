import json
from datasets import load_dataset
from tqdm.auto import tqdm


def save_mmlu_data(output_file: str):
    """Loads all splits of the MMLU dataset and saves questions, subjects, choices, and answers to a single JSON file."""
    splits = ["test", "auxiliary_train", "dev", "validation"]

    mmlu_data = {}
    
    for split in splits:
        dataset = load_dataset("cais/mmlu", "all", split=split, cache_dir=cache_dir)
        
        for example in tqdm(dataset, desc=f"Processing MMLU {split} split"):
            mmlu_data[example["question"].strip()] = dataset.features["answer"].int2str(example["answer"])
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mmlu_data, f, ensure_ascii=False, indent=4)


# Define files
mmlu_output_file = "all_mmlu_splits.json"

# Save MMLU data
save_mmlu_data(mmlu_output_file)


