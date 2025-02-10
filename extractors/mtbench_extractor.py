import pandas as pd
import ast
import json


def read_mtbench():
    routerbench = pd.read_pickle(
        "/Users/samikd/research/3p-src/routerbench/data/routerbench/input_wide__01-16-10__routerbench.pkl"
    )

    mtbench = routerbench[
        routerbench["eval_name"].isin(["mtbench-math", "mtbench-reference", "mtbench"])
    ][
        [
            "prompt",
            "gpt-3.5-turbo-1106",
            "gpt-4-1106-preview",
            "gpt-3.5-turbo-1106|model_response",
            "gpt-4-1106-preview|model_response",
        ]
    ]

    mtbench["first_turn_prompt"] = mtbench["prompt"].apply(
        lambda conversation: ast.literal_eval(conversation)[0]
    )
    mtbench["second_turn_prompt"] = mtbench["prompt"].apply(
        lambda conversation: ast.literal_eval(conversation)[1]
    )

    mtbench["first_turn_response_gpt-3.5-turbo-1106"] = mtbench[
        "gpt-3.5-turbo-1106|model_response"
    ].apply(lambda conversation: ast.literal_eval(conversation)[0])

    mtbench["second_turn_response_gpt-3.5-turbo-1106"] = mtbench[
        "gpt-3.5-turbo-1106|model_response"
    ].apply(lambda conversation: ast.literal_eval(conversation)[1])

    mtbench["first_turn_response_gpt-4-1106-preview"] = mtbench[
        "gpt-4-1106-preview|model_response"
    ].apply(lambda conversation: ast.literal_eval(conversation)[0])

    mtbench["second_turn_response_gpt-4-1106-preview"] = mtbench[
        "gpt-4-1106-preview|model_response"
    ].apply(lambda conversation: ast.literal_eval(conversation)[1])

    mtbench["qdrant_key"] = (
        mtbench["first_turn_prompt"] + "\n" + mtbench["second_turn_prompt"]
    )

    mtbench_train = mtbench.sample(frac=0.75, random_state=42)
    mtbench_train[["qdrant_key", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"]].to_csv(
        "data/routerbench_mtbench_train.csv", index=True, header=True
    )

    mtbench_test = mtbench.drop(mtbench_train.index)
    mtbench_test[["qdrant_key", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"]].to_csv(
        "data/routerbench_mtbench_test.csv", index=True, header=True
    )

    # Write scores to CSV
    scores_df = mtbench[["qdrant_key", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"]]
    scores_df.to_csv("data/routerbench_mtbench.csv", index=True, header=True)

    return mtbench.drop(
        columns=[
            "prompt",
            "gpt-3.5-turbo-1106|model_response",
            "gpt-4-1106-preview|model_response",
        ]
    )


def write_mtbench(mtbench, llm):
    # Create conversation records in the desired format
    conversations = []
    for idx, row in mtbench.iterrows():
        conversation = {
            "conversation_id": f"{idx}",
            "messages": [
                {"role": "user", "content": row["first_turn_prompt"]},
                {
                    "role": "assistant",
                    "content": row[f"first_turn_response_{llm}"],
                },
                {"role": "user", "content": row["second_turn_prompt"]},
                {
                    "role": "assistant",
                    "content": row[f"second_turn_response_{llm}"],
                },
            ],
            "optimal_llm": llm,
        }
        conversations.append(conversation)

    # Write to JSONL file
    import json

    output_path = f"data/mtbench_{llm}.jsonl"
    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")


if __name__ == "__main__":
    mtbench = read_mtbench()
    write_mtbench(mtbench, "gpt-3.5-turbo-1106")
    write_mtbench(mtbench, "gpt-4-1106-preview")
