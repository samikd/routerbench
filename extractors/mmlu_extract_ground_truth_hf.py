import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from tqdm.auto import tqdm


def setup_logging(log_level: int = logging.INFO) -> None:
    """Configure logging with a standard format.
    
    Args:
        log_level: The logging level to use. Defaults to logging.INFO.
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def extract_mmlu_ground_truth(
    output_file: str,
    cache_dir: Optional[str] = None,
    splits: Optional[List[str]] = None
) -> Dict[str, str]:
    """Extract ground truth answers from all splits of the MMLU dataset.
    
    Args:
        output_file: Path where the JSON output file will be saved.
        cache_dir: Optional directory for caching the downloaded dataset.
        splits: Optional list of splits to process. Defaults to ["test", "auxiliary_train", "dev", "validation"].
    
    Returns:
        Dict mapping questions to their correct answers.
        
    Raises:
        Exception: If there's an error loading the dataset or saving the output.
    """
    logger = logging.getLogger(__name__)
    
    if splits is None:
        splits = ["test", "auxiliary_train", "dev", "validation"]
    
    logger.info(f"Starting MMLU ground truth extraction for splits: {splits}")
    mmlu_data = {}
    
    try:
        for split in splits:
            logger.info(f"Loading MMLU {split} split...")
            dataset = load_dataset("cais/mmlu", "all", split=split, cache_dir=cache_dir)
            
            for example in tqdm(dataset, desc=f"Processing MMLU {split} split"):
                question = example["question"].strip()
                answer = dataset.features["answer"].int2str(example["answer"])
                mmlu_data[question] = answer
            
            logger.info(f"Completed processing {split} split. Current total questions: {len(mmlu_data)}")
        
        # Ensure the output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the data
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mmlu_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Successfully saved {len(mmlu_data)} question-answer pairs to {output_file}")
        return mmlu_data
    
    except Exception as e:
        logger.error(f"Error during MMLU ground truth extraction: {str(e)}")
        raise


def main():
    """Main entry point of the script."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Define output file
        mmlu_output_file = "all_mmlu_splits.json"
        
        # Extract and save MMLU ground truth data
        extract_mmlu_ground_truth(mmlu_output_file)
        
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()


