import json
import logging
from pathlib import Path
from typing import List, Dict, Set
import glob

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


def load_mmlu_ground_truth(mmlu_data_file: str) -> Dict[str, str]:
    """Load MMLU ground truth data from JSON file.
    
    Args:
        mmlu_data_file: Path to the MMLU ground truth JSON file.
        
    Returns:
        Dictionary mapping questions to their ground truth answers.
        
    Raises:
        FileNotFoundError: If the MMLU data file doesn't exist.
        json.JSONDecodeError: If the MMLU data file is not valid JSON.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading MMLU ground truth data from {mmlu_data_file}")
    
    try:
        with open(mmlu_data_file, "r", encoding="utf-8") as f:
            mmlu_data = json.load(f)
        logger.info(f"Successfully loaded {len(mmlu_data)} ground truth entries")
        return mmlu_data
    except FileNotFoundError:
        logger.error(f"MMLU ground truth file not found: {mmlu_data_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in MMLU ground truth file: {str(e)}")
        raise


def get_input_files(data_dir: str, exclude_files: Set[str]) -> List[Path]:
    """Get list of input JSONL files from directory, excluding specified files.
    
    Args:
        data_dir: Directory containing JSONL files.
        exclude_files: Set of filenames to exclude.
        
    Returns:
        List of Path objects for JSONL files to process.
    """
    logger = logging.getLogger(__name__)
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    input_files = [
        Path(f) for f in glob.glob(str(data_path / "*.jsonl"))
        if Path(f).name not in exclude_files
    ]
    
    logger.info(f"Found {len(input_files)} JSONL files to process")
    return input_files


def process_jsonl_file(
    input_file: Path,
    mmlu_data: Dict[str, str],
    output_dir: Path
) -> None:
    """Process a single JSONL file and add ground truth data.
    
    Args:
        input_file: Path to input JSONL file.
        mmlu_data: Dictionary of MMLU ground truth data.
        output_dir: Directory to save output files.
        
    Raises:
        IOError: If there are issues reading/writing files.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing file: {input_file}")
    
    output_file = output_dir / f"{input_file.stem}_with_gt.jsonl"
    updated_data = []
    unknown_count = 0
    total_count = 0
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Processing {input_file.name}"):
                total_count += 1
                try:
                    sample = json.loads(line)
                    if "messages" in sample and sample["messages"]:
                        user_content = sample["messages"][0]["content"]
                        questions = user_content.split("\n\nPlease answer with the letter of the correct answer.\n\n")
                        last_question = questions[-1].split("\nA)")[0].strip()
                        
                        ground_truth = mmlu_data.get(last_question, "Unknown")
                        if ground_truth == "Unknown":
                            unknown_count += 1
                        
                        sample["ground_truth"] = ground_truth
                    updated_data.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line in {input_file}: {str(e)}")
                    continue
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write output file
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in updated_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        logger.info(f"File {input_file.name} processed:")
        logger.info(f"- Total entries: {total_count}")
        logger.info(f"- Unknown ground truth: {unknown_count}")
        logger.info(f"- Output saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {str(e)}")
        raise


def main():
    """Main entry point of the script."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Configuration
        data_dir = "data"
        output_dir = Path("data/processed")
        mmlu_data_file = "all_mmlu_splits.json"
        exclude_files = {"mmlu_train.jsonl", "mmlu_test.jsonl"}
        
        # Load MMLU ground truth data
        mmlu_data = load_mmlu_ground_truth(mmlu_data_file)
        
        # Get input files
        input_files = get_input_files(data_dir, exclude_files)
        
        if not input_files:
            logger.warning("No JSONL files found to process")
            return
        
        # Process each input file
        for input_file in input_files:
            try:
                process_jsonl_file(input_file, mmlu_data, output_dir)
            except Exception as e:
                logger.error(f"Failed to process {input_file}: {str(e)}")
                continue
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

