import json
import os
import random
from collections import Counter
import argparse
import numpy as np
from datasets import load_dataset
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
    system_prompt= "You are an intelligent and objective assistant. Carefully read each statement and determine whether it is morally True (label 1) or False (label 0). Do not provide reasoning and output **ONLY** the label: 0 (False) or 1 (True), NOT any other symbol"
    for i in train:
        i['source'] = 'moral_dataset'
        i['system_prompt'] = system_prompt
        i['user_prompt'] = f"Claim: {i['claim']}\nI think this claim is"

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

def predict_label(client, model, example):
    full_prompt = f"{example['system_prompt']}\n{example['user_prompt']}"
    print(full_prompt)
    response = client.completions.create(
        model=model,
        prompt=full_prompt,
        max_tokens=5,
        temperature=0
    )
    print(score)
    score = response.choices[0].text.strip().lower()
    print('score is:',score)
    
    # Normalize the output
    if score in ["1", "true", "yes" , "correct" , "reasonable" , "acceptable", "valid" , 1]:
        return 1
    elif score in ["0", "false", "no" , "wrong", "unreasonable" , "unacceptable", "invalid" , 0]:
        return 0
    else:
        print("⚠️ Unexpected model output:", score)
        return -1


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

    # OpenAI/vLLM client (pointing to local vLLM server)
    client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")

    print("Initial label distribution:", Counter([v['label'] for v in demonstrations.values() if v['label'] is not None]))
    print("Initial accuracy:", calculate_accuracy(demonstrations))

    # Predict labels for all examples without seeds
    for uid, example in demonstrations.items():
        if example["label"] is None:
            example["label"] = predict_label(client, args.model, example)

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
    from core.utils import setup_environment
    setup_environment(logger_level="error")
    args = get_args()
    random.seed(args.seed)
    main(args)
