import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from routers.common import TOKEN_COSTS
from convertors.common import calculate_cost_for_prompt_and_response, get_highest_accuracy_lowest_cost
import tiktoken

# Define the models we want to analyze
MODELS_TO_ANALYZE = [
    "mistral/open-mistral-7b",
    "anyscale/HuggingFaceH4/zephyr-7b-beta",
    "deepinfra/cognitivecomputations/dolphin-2.6-mixtral-8x7b",
    "replicate/meta/llama-3-8b"
]

# Map between model names in MODELS_TO_ANALYZE and score column names in JSON
MODEL_NAME_MAPPING = {
    "mistral/open-mistral-7b": "mistralai/Mistral-7B-v0.1",
    "anyscale/HuggingFaceH4/zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",
    "deepinfra/cognitivecomputations/dolphin-2.6-mixtral-8x7b": "cognitivecomputations/dolphin-2.6-mistral-7b",
    "replicate/meta/llama-3-8b": "meta-llama/Meta-Llama-3-8B"
}

def count_tokens(text):
    """Count the number of tokens in the text using GPT tokenizer."""
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

def calculate_prompt_cost(prompt, model_name):
    """Calculate the cost for processing a prompt with a specific model."""
    if model_name not in TOKEN_COSTS:
        raise ValueError(f"No cost information available for model: {model_name}")
    
    num_tokens = count_tokens(prompt)
    prompt_cost_per_token = TOKEN_COSTS[model_name]["prompt"]
    return num_tokens * prompt_cost_per_token

def clean_answer(answer):
    """Clean the answer for comparison."""
    return str(answer).strip().lower()

def calculate_accuracy(responses, ground_truths):
    """Calculate accuracy for a set of responses."""
    if len(responses) != len(ground_truths):
        raise ValueError("Number of responses and ground truths must match")
    
    correct = 0
    total = 0
    for resp, gt in zip(responses, ground_truths):
        if pd.isna(resp) or pd.isna(gt):
            continue
        if clean_answer(resp) == clean_answer(gt):
            correct += 1
        total += 1
    return (correct / total) * 100 if total > 0 else 0

def load_mmlu_data(file_path):
    """Load and process the MMLU test data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    print("\nLoaded data structure:")
    print("Number of rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("\nSample of first row:")
    print(df.iloc[0])
    
    return df

def process_data(df, models):
    """Calculate costs and accuracies for each model."""
    print("Processing data for each model...")
    
    results = {
        'costs': {},
        'accuracies': {},
        'oracle_stats': {}
    }
    
    # Calculate costs and prepare accuracy data
    for model in tqdm(models, desc="Processing models"):
        # Get the corresponding JSON model name
        json_model_name = MODEL_NAME_MAPPING[model]
        
        # Calculate prompt costs using the MODELS_TO_ANALYZE names (for TOKEN_COSTS lookup)
        df[f"{model}_prompt_cost"] = df["question"].apply(
            lambda prompt: calculate_prompt_cost(prompt, model)
        )
        
        # Convert scores to binary (1 if score > 0, 0 otherwise)
        df[model] = df['scores'].apply(
            lambda scores_dict: 1 if scores_dict.get(json_model_name, 0) > 0 else 0
        )
        #print(df[model].head())
        #print(df[f"{model}_prompt_cost"].head())
        #x = 1/0

    
        # Calculate accuracy as percentage of non-zero scores
        accuracy = df[model].mean() * 100
        results['accuracies'][model] = accuracy
        print(f"Model {model} accuracy: {accuracy:.2f}% (based on binary scores)")
        
        # Store total cost (just prompt cost since we don't need response costs)
        total_cost = df[f"{model}_prompt_cost"].sum()
        results['costs'][model] = total_cost
    
    # Ensure all required columns exist for get_highest_accuracy_lowest_cost
    for model in models:
        if model not in df.columns:
            print(f"Warning: Score column {model} not found, initializing with zeros")
            df[model] = 0
        if f"{model}_prompt_cost" not in df.columns:
            print(f"Warning: Cost column {model}_prompt_cost not found, initializing with zeros")
            df[f"{model}_prompt_cost"] = 0
    
    # Use get_highest_accuracy_lowest_cost for oracle selection
    print("Calculating oracle selections...")
    print("\nScore distribution before oracle selection:")
    for model in models:
        value_counts = df[model].value_counts()
        print(f"\n{model}:")
        print(value_counts)
    
    df['oracle_model'] = df.apply(
        lambda row: get_highest_accuracy_lowest_cost(row, models),
        axis=1
    )
    
    # Calculate oracle cost using prompt costs
    df['oracle_cost'] = df.apply(
        lambda row: row[f"{row['oracle_model']}_prompt_cost"],
        axis=1
    )
    
    # Calculate oracle statistics
    oracle_selections = df['oracle_model'].value_counts()
    total_prompts = len(df)
    
    results['oracle_stats'] = {
        'total_prompts': total_prompts,
        'model_selections': {
            model: {
                'count': int(oracle_selections.get(model, 0)),
                'percentage': float(oracle_selections.get(model, 0) / total_prompts * 100)
            }
            for model in models
        }
    }
    
    # Calculate total oracle cost
    results['oracle_stats']['total_cost'] = df['oracle_cost'].sum()
    
    return df, results

def analyze_results(df, results):
    """Analyze and print cost and accuracy statistics."""
    print("\nAnalysis Results:")
    print("-" * 50)
    
    # Print costs
    print("\nTotal Prompt Costs per Model:")
    for model, cost in results['costs'].items():
        print(f"{model}: ${cost:.6f}")
    
    # Print accuracies if available
    if results['accuracies']:
        print("\nAccuracies per Model:")
        for model, accuracy in results['accuracies'].items():
            print(f"{model}: {accuracy:.2f}%")
    
    # Print average cost per prompt
    print("\nAverage Cost per Prompt:")
    for model in MODELS_TO_ANALYZE:
        avg_cost = df[f"{model}_prompt_cost"].mean()
        print(f"{model}: ${avg_cost:.8f}")
    
    # Print oracle statistics
    print("\nOracle (Highest Accuracy & Lowest Cost) Statistics:")
    print(f"Total Oracle Cost: ${results['oracle_stats']['total_cost']:.6f}")
    print("\nModel Selection Frequency:")
    for model, stats in results['oracle_stats']['model_selections'].items():
        print(f"{model}: {stats['count']} times ({stats['percentage']:.2f}%)")

def analyze_cost_savings(results):
    """Analyze cost savings from oracle routing."""
    # Get total costs for each model
    model_costs = results['costs']
    oracle_cost = results['oracle_stats']['total_cost']
    
    # Create comparison data
    cost_comparison = []
    for model, cost in model_costs.items():
        savings = cost - oracle_cost
        savings_percentage = (savings / cost) * 100 if cost > 0 else 0
        cost_comparison.append({
            'model': model,
            'total_cost': cost,
            'savings_vs_oracle': savings,
            'savings_percentage': savings_percentage
        })
    
    # Add oracle row
    cost_comparison.append({
        'model': 'oracle',
        'total_cost': oracle_cost,
        'savings_vs_oracle': 0,
        'savings_percentage': 0
    })
    
    # Convert to DataFrame and save
    cost_df = pd.DataFrame(cost_comparison)
    cost_df.to_csv('cost_comparison.csv', index=False)
    
    # Print analysis
    print("\nCost Analysis:")
    print("-" * 50)
    print(f"\nOracle Total Cost: ${oracle_cost:.6f}")
    print("\nCost Comparison vs Oracle:")
    for row in cost_comparison[:-1]:  # Exclude oracle row from comparison
        print(f"\n{row['model']}:")
        print(f"  Total Cost: ${row['total_cost']:.6f}")
        print(f"  Savings: ${row['savings_vs_oracle']:.6f}")
        print(f"  Savings Percentage: {row['savings_percentage']:.2f}%")
    
    return cost_df

def main():
    # Load MMLU data
    mmlu_file = Path("test_mmlu_with_subjects.json")
    if not mmlu_file.exists():
        mmlu_file = Path("routerbench/test_mmlu_with_subjects.json")
    
    print(f"Loading data from {mmlu_file}")
    df = load_mmlu_data(mmlu_file)
    
    # Process data
    df, results = process_data(df, MODELS_TO_ANALYZE)
    
    # Debug prints
    print("\nVerifying columns in dataframe:")
    print("All columns:", df.columns.tolist())
    print("\nFirst few rows of oracle columns:")
    print(df[['oracle_model', 'oracle_cost']].head())
    
    # Analyze results
    analyze_results(df, results)
    
    # Analyze cost savings and create comparison CSV
    cost_df = analyze_cost_savings(results)
    print("\nCost comparison saved to cost_comparison.csv")
    
    # Save detailed results
    output_file = "mmlu_analysis_results.csv"
    print(f"\nSaving detailed results to {output_file}")
    
    # Ensure oracle columns are at the front for visibility
    columns = df.columns.tolist()
    oracle_columns = ['oracle_model', 'oracle_cost']
    other_columns = [col for col in columns if col not in oracle_columns]
    df = df[oracle_columns + other_columns]
    
    df.to_csv(output_file, index=False)
    print(f"Saved file with {len(df.columns)} columns")
    print("First few columns:", df.columns[:5].tolist())
    
    # Save summary
    summary = {
        'costs': results['costs'],
        'accuracies': results['accuracies'],
        'oracle_stats': results['oracle_stats']
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("mmlu_summary.csv")
    print("Summary saved to mmlu_summary.csv")

if __name__ == "__main__":
    main() 