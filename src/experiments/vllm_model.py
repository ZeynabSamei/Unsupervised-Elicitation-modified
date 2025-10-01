import json
import os
import random
from collections import Counter
import argparse
import numpy as np
from src.tools.path_utils import get_root_directory
from openai import OpenAI

# ----------------------------
# Helper functions
# ----------------------------

def load_data(args):
    """
    Load moral_dataset and prepare prompts.
    """
    data_path = get_root_directory() / "data/train_moral_dataset.json"
    with open(data_path) as f:
        train = json.load(f)

    # System prompt text
    system_prompt = (
        "You are an intelligent and objective assistant. Carefully read each statement and determine whether it is morally True (label 1) or False (label 0).Do not provide reasoning and output ONLY the label: 0 (False) or 1 (True)."
    )

    for i in train:
        i['source'] = 'moral_dataset'
        i['system_prompt'] = system_prompt
        i['user_prompt'] = f"{system_prompt}\nClaim: {i['claim']}\nLabel:"

    return train


def predict_label(client, model, example):
    """
    Predict label using OpenAI/vLLM client.
    """
    response = client.completions.create(
        model=model,
        prompt=example["user_prompt"],
        max_tokens=1,
        temperature=0
    )

    score = response.choices[0].text.strip().lower()

    # Normalize the output
    if score in ["1", "true", "yes", "correct"]:
        return 1
    elif score in ["0", "false", "no", "wrong"]:
        return 0
    else:
        print("⚠️ Unexpected model output:", score)
        return score


# ----------------------------
# Main
# ----------------------------

def main(args):
    # Load data
    train = load_data(args)

    # OpenAI/vLLM client (pointing to local vLLM server)
    client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")

    # Predict labels for all examples
    for example in train:
        example["label"] = predict_label(client, args.model, example)

    print("Label distribution:", Counter([v['label'] for v in train]))

    # Save results
    save_path = "/home/maliza/scratch/results/moral_dataset_results.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(train, f, indent=2)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    return parser.parse_args()


if __name__ == "__main__":
    from core.utils import setup_environment
    setup_environment(logger_level="error")
    args = get_args()
    random.seed(args.seed)
    main(args)
