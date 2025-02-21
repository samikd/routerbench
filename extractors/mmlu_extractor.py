import pandas as pd
import ast
from sklearn.model_selection import train_test_split


def read_mmlu(file_path):
    routerbench = pd.read_pickle(file_path)

    mmlu = routerbench[
        routerbench["eval_name"].isin(
            [
                "mmlu-abstract-algebra",
                "mmlu-anatomy",
                "mmlu-astronomy",
                "mmlu-business-ethics",
                "mmlu-clinical-knowledge",
                "mmlu-college-biology",
                "mmlu-college-chemistry",
                "mmlu-college-computer-science",
                "mmlu-college-mathematics",
                "mmlu-college-medicine",
                "mmlu-college-physics",
                "mmlu-computer-security",
                "mmlu-conceptual-physics",
                "mmlu-econometrics",
                "mmlu-electrical-engineering",
                "mmlu-elementary-mathematics",
                "mmlu-formal-logic",
                "mmlu-global-facts",
                "mmlu-high-school-biology",
                "mmlu-high-school-chemistry",
                "mmlu-high-school-computer-science",
                "mmlu-high-school-european-history",
                "mmlu-high-school-geography",
                "mmlu-high-school-government-and-politics",
                "mmlu-high-school-macroeconomics",
                "mmlu-high-school-mathematics",
                "mmlu-high-school-microeconomics",
                "mmlu-high-school-physics",
                "mmlu-high-school-psychology",
                "mmlu-high-school-statistics",
                "mmlu-high-school-us-history",
                "mmlu-high-school-world-history",
                "mmlu-human-aging",
                "mmlu-human-sexuality",
                "mmlu-international-law",
                "mmlu-jurisprudence",
                "mmlu-logical-fallacies",
                "mmlu-machine-learning",
                "mmlu-management",
                "mmlu-marketing",
                "mmlu-medical-genetics",
                "mmlu-miscellaneous",
                "mmlu-moral-disputes",
                "mmlu-moral-scenarios",
                "mmlu-nutrition",
                "mmlu-philosophy",
                "mmlu-prehistory",
                "mmlu-professional-accounting",
                "mmlu-professional-law",
                "mmlu-professional-medicine",
                "mmlu-professional-psychology",
                "mmlu-public-relations",
                "mmlu-security-studies",
                "mmlu-sociology",
                "mmlu-us-foreign-policy",
                "mmlu-virology",
                "mmlu-world-religions",
            ]
        )
    ][
        [
            "prompt",
            "gpt-3.5-turbo-1106",
            "gpt-4-1106-preview",
            "gpt-3.5-turbo-1106|model_response",
            "gpt-4-1106-preview|model_response",
        ]
    ]

    mmlu["first_turn_prompt"] = mmlu["prompt"].apply(
        lambda conversation: ast.literal_eval(conversation)[0]
    )

    mmlu["first_turn_response_gpt-3.5-turbo-1106"] = mmlu[
        "gpt-3.5-turbo-1106|model_response"
    ].apply(lambda conversation: ast.literal_eval(conversation)[0])

    mmlu["first_turn_response_gpt-4-1106-preview"] = mmlu[
        "gpt-4-1106-preview|model_response"
    ].apply(lambda conversation: ast.literal_eval(conversation)[0])

    return mmlu.drop(
        columns=[
            "prompt",
            "gpt-3.5-turbo-1106|model_response",
            "gpt-4-1106-preview|model_response",
        ]
    )


def split_mmlu(mmlu):
    mmlu_slice = mmlu.sample(n=300, random_state=42)

    train = mmlu_slice.sample(frac=2.0 / 3.0, random_state=42)
    test = mmlu_slice.drop(train.index)

    print(f"Train: {train.shape}, Test: {test.shape}.")

    return train, test


def write_mmlu(mmlu, llm, suffix):
    # Create conversation records in the desired format
    conversations = []
    for idx, row in mmlu.iterrows():
        conversation = {
            "conversation_id": f"{idx}",
            "messages": [
                {"role": "user", "content": row["first_turn_prompt"]},
                {
                    "role": "assistant",
                    "content": row[f"first_turn_response_{llm}"],
                },
            ],
            "optimal_llm": llm,
        }
        conversations.append(conversation)

    # Write to JSONL file
    import json

    output_path = f"data/mmlu_{llm}_{suffix}.jsonl"
    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")


def write_mmlu_test(mmlu):
    pass


def write_mmlu_train(mmlu, llm):
    # Create conversation records in the desired format
    conversations = []
    for idx, row in mmlu.iterrows():
        conversation = {
            "conversation_id": f"{idx}",
            "messages": [
                {"role": "user", "content": row["first_turn_prompt"]},
                {
                    "role": "assistant",
                    "content": row[f"first_turn_response_{llm}"],
                },
            ],
            "optimal_llm": row["optimal_llm"],
        }
        conversations.append(conversation)

    # Write to JSONL file
    import json

    output_path = f"data/mmlu_train.jsonl"
    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")


if __name__ == "__main__":
    mmlu = read_mmlu("data/routerbench/input_wide__01-16-10__routerbench.pkl")

    mmlu["optimal_llm"] = mmlu[["gpt-3.5-turbo-1106", "gpt-4-1106-preview"]].idxmax(
        axis=1
    )

    train, test = split_mmlu(mmlu)

    print(mmlu.columns)

    write_mmlu_train(train, "gpt-4-1106-preview") # FIXME

    write_mmlu(train, "gpt-4-1106-preview", suffix="train")
    write_mmlu(train, "gpt-3.5-turbo-1106", suffix="train")

    write_mmlu(test, "gpt-4-1106-preview", suffix="test")
    write_mmlu(test, "gpt-3.5-turbo-1106", suffix="test")
