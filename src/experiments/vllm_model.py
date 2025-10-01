import json
import os
import random
from collections import Counter
from copy import deepcopy
from datasets import load_dataset
import argparse

from core.llm_api.llm import ModelAPI
from core.utils import setup_environment
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
    system_prompt = "You are an intelligent and objective assistant. Carefully read each statement and determine whether it is morally True (label 1) or False (label 0). Do not provide your reasoning and just explicitly assign the appropriate label."
    

    for i in train:
        i['source'] = 'moral_dataset'
        i['consistency_key'] = 'A' if i['label'] else 'B'
        i['prompt'] = f"{system_prompt}\nClaim: {i['claim']}\nI think this claim is"

    # sample batch
    fewshot_ids = random.sample(range(len(train)), args.batch_size)
    return train, fewshot_ids


def initialize(train, fewshot_ids, args):
    """
    Initialize demonstration dict and seed labels.
    """
    demonstrations = {}
    random_init_labels = [1] * (args.num_seed // 2) + [0] * (args.num_seed // 2)
    random.shuffle(random_init_labels)

    for id, i in enumerate(fewshot_ids):
        item = train[i]
        item["vanilla_label"] = item["label"]
        item["uid"] = id
        if id >= args.num_seed:
            item["label"] = None
            item["type"] = "predict"
        else:
            item["label"] = random_init_labels[id]
            item["type"] = "seed"
        demonstrations[id] = item

    return demonstrations


def predict_label(model, example):
    """
    Use base model to predict label.
    """
    client=OpenAI(api_key='EMPTY', base_url='http://127.0.0.1:8000/v1')
    response = client.completions.create(
        prompt=example["prompt"],
        logprobs=20,
        max_tokens=1,
        model=model
    )
    # Here assume response[0]["score"] exists; adjust if ModelAPI returns differently
    score = choices[0]["text"]
    return score


def calculate_accuracy(demonstrations):
    labels = [v["label"] for v in demonstrations.values() if v["label"] is not None]
    vanilla_labels = [v["vanilla_label"] for v in demonstrations.values() if v["label"] is not None]
    return np.mean([l == vl for l, vl in zip(labels, vanilla_labels)])


# ----------------------------
# Main
# ----------------------------

def main(args):
    train, fewshot_ids = load_data(args)
    demonstrations = initialize(train, fewshot_ids, args)

    # Initialize ModelAPI
    model_api = ModelAPI(
        anthropic_num_threads=20,
        openai_fraction_rate_limit=0.99
    )

    print("Initial label distribution:", Counter([v['label'] for v in demonstrations.values() if v['label'] is not None]))
    print("Initial accuracy:", calculate_accuracy(demonstrations))

    # Predict labels for all examples without seeds
    for uid, example in demonstrations.items():
        if example["label"] is None:
            example["label"] = predict_label(model_api, example)

    print("Final label distribution:", Counter([v['label'] for v in demonstrations.values()]))
    print("Final accuracy:", calculate_accuracy(demonstrations))

    # Save results
    save_path = "/home/maliza/scratch/results/moral_dataset_results.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(demonstrations, f, indent=2)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_seed", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    return parser.parse_args()


if __name__ == "__main__":
    import numpy as np
    setup_environment(logger_level="error")
    args = get_args()
    random.seed(args.seed)
    main(args)
