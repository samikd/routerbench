# RouterBench

### [Paper](https://arxiv.org/abs/2403.12031) | [Dataset](https://huggingface.co/datasets/withmartian/routerbench)

The code for the paper ROUTERBENCH: A Benchmark for Multi-LLM Routing System

## Setup process
1. Create .env file in the root directory. With the following variables:
```
CONNECTION_STRING='your mongodb connection string'
```
if you want to use MongoDB as an embedding cache. 

We use Martian as it provides a unified gateway to access all the models we use. Please visit withmartian.com to create a new account and get started.

2. In root directory, run `pip install -e .` to install the packages.

## Running the pipeilne 

The pipeline relies on various command line arguments to specify the configuration. Alternatively, you can specify the 
configuration in a yaml file and pass it to the command line. Example configurations are in the `configs/` directory.

First, if desired, make sure there is a MongoDB instance running that you can connect to. If there is not one, ensure that `local_cache: true` to ensure that 
the code only uses local files for caching.

Second, run `convert_data.py --config=configs/convert_data.yaml` to process the different data formats into a common format.
    This script can take raw format from `martian-evals` repo, as well as other relevant input formats.

Third, run `evaluate_routers.py --config=configs/evaluate_routers.yaml` to use the processed data to evaluate different routers. It generates a csv file (long format) with the results of the evaluation, and creates an EvaluationCollection containing the results.

Fourth, run `visualize_results.py --config=configs/visualize.yaml` uses the EvaluationCollection to visualize the results in a performance-vs-cost plot.

For these configurations, the paths to the data files will need to be updated to use your local paths. Example files to recreate results from the paper are available on [Hugging Face](https://huggingface.co/datasets/withmartian/routerbench).


## Contribution Guide

The code is designed to be easily extended. To add a new router, or convertor for a different input data format, simply look
at the abstract classes `AbstractRouter` and `AbstractConvertor` in `routers/` and `convertors/` respectively.

- For each PR, please run flake8, black, isort
```bash
flake8 $(git ls-files '*.py')
black $(git ls-files '*.py')
isort $(git ls-files '*.py')
```
`$(git ls-files '*.py')` is for running only the files tracked by git, so exclude virtual env files or data files.
You may need to run `pip install flake8 black isort` if you don't have them installed.


# MMLU Ground Truth Extractor

This script extracts ground truth answers from the MMLU (Massive Multitask Language Understanding) dataset using the Hugging Face datasets library.

## Features

- Extracts questions and answers from all MMLU dataset splits
- Saves output in JSON format

## Requirements

```bash
pip install datasets tqdm
```

## Usage

### As a Script

1. Simple usage with default settings:
```bash
python extract_mmlu_hf.py
```

This will:
- Process all splits (test, auxiliary_train, dev, validation)
- Save the output to `all_mmlu_splits.json`
- Use default cache directory for dataset downloads

### As a Module

```python
from extract_mmlu_hf import extract_mmlu_ground_truth, setup_logging

# Set up logging (optional)
setup_logging()

# Basic usage
data = extract_mmlu_ground_truth("all_mmlu_splits.json")

# Advanced usage with custom parameters
data = extract_mmlu_ground_truth(
    output_file="custom_output.json",
    cache_dir="/path/to/cache",
    splits=["test", "validation"]  # Only process specific splits
)
```

## Output Format

The script generates a JSON file with the following structure:

```json
{
    "question1": "answer1",
    "question2": "answer2",
    ...
}
```

## Modal update Guide
To deploy the updated modal app, run the following commands:
```bash
modal deploy modal_router.py
```

## MMLU Ground Truth Processing

The `add_ground_truth_mmlu.py` script processes JSONL files containing MMLU evaluations and adds ground truth answers to them.

### Features
- Processes multiple JSONL files in batch
- Excludes specified files (e.g., training and test sets)
- Comprehensive error handling and logging
- Progress tracking with detailed statistics
- Organizes processed files in a separate directory

### Usage

1. Setup:
   - Place your JSONL files in the `data/` directory
   - Ensure `all_mmlu_splits.json` (ground truth data) is in the root directory

2. Run the script:
```bash
python add_ground_truth_mmlu.py
```

3. Output:
   - Processed files will be saved in `data/processed/` directory
   - Each output file will have "_with_gt" suffix
   - Detailed logs will show processing statistics and any issues

### Configuration
The script can be configured by modifying these variables in `main()`:
```python
data_dir = "data"  # Input directory containing JSONL files
output_dir = Path("data/processed")  # Output directory for processed files
mmlu_data_file = "all_mmlu_splits.json"  # Ground truth data file
exclude_files = {"mmlu_train.jsonl", "mmlu_test.jsonl"}  # Files to skip
```

## Citation
If you use this code, please cite the following paper:
```bibtex
@article{hu2024routerbench,
  title   = {ROUTERBENCH: A Benchmark for Multi-LLM Routing System},
  author  = {Qitian Jason Hu and Jacob Bieker and Xiuyu Li and Nan Jiang and Benjamin Keigwin and Gaurav Ranganath and Kurt Keutzer and Shriyash Kaustubh Upadhyay},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2403.12031}
}
```


