# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import random
import os
from collections import Counter
import numpy as np
import requests
from pathlib import Path

# ---------------------
# Utility functions
# ---------------------
def get_root_directory():
    return Path(__file__).resolve().parent.parent.parent  # adjust to repo root

def load_data(args):
    if args.testbed == "moral_dataset":
        with open(get_root_directory() / "data/train_moral_dataset.json") as f:
            train = json.load(f)
        template = """Claim: {claim}\nI think this claim is **** """

        for i in train:
            i['source'] = 'moral_dataset'
            i['consistency_key'] = 'A' if i['label'] else 'B'
            i['prompt'] = template.format(claim=i['claim'])
        args.GROUP_SIZE = 1

        # Optional: add second dataset mapping here if needed

    # Group by consistency_id
    train_map = {}
    for i in train:
        cid = i.get('consistency_id', random.randint(0, 10000))  # fallback
        if cid not in train_map:
            train_map[cid] = []
        train_map[cid].append(i)
    
    train = [item for group in train_map.values() for item in group]

    # sample a batch of batch_size datapoints
    fewshot_ids = random.sample(
        list(range(len(train)// args.GROUP_SIZE)), args.batch_size // args.GROUP_SIZE
    )
    fewshot_ids = [
        i * args.GROUP_SIZE + j for i in fewshot_ids for j in range(args.GROUP_SIZE)
    ]

    return train, fewshot_ids

def calculate_accuracy(train_data, inconsistent_pairs):
    train_probs = []
    for i in train_data.values():
        if i.get("label") is None:
            continue
        train_probs.append(i["score"] if i["label"] == 1 else -i["score"])
    train_prob = np.mean(train_probs) if train_probs else 0

    save_path = get_root_directory() / "results/final_train_data.json"
    os.makedirs(save_path.parent, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(train_data, f, indent=2)

    return {
        "train_accuracy": 0 if len(train_data) == 0 else np.mean(
            [i["label"] == i.get("vanilla_label") for i in train_data.values()]
        ),
        "train_label_distribution": Counter([i.get("vanilla_label") for i in train_data.values()]),
        "train_predict_distribution": Counter([i.get("label") for i in train_data.values()]),
        "train_prob": train_prob,
        "train_size": len(train_data),
        "inconsistent_num": len(inconsistent_pairs),
    }

# ---------------------
# Main client
# ---------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Client for vLLM API server")
    parser.add_argument("--stream", action="store_true", help="Enable streaming response")
    parser.add_argument("--testbed", type=str, default="moral_dataset")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    return parser.parse_args()

def main(args):
    train, fewshot_ids = load_data(args)

    chat_template = {"system": "{system}", "user": "{user}", "assistant": "{assistant}"}
    api_url = "http://localhost:8000/v1/chat/completions"

    for idx in fewshot_ids:
        prompt = train[idx]["prompt"]
        data = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": "You are an intelligent and objective assistant. Carefully read each statement and determine whether it is morally True (label 1) or False (label 0). Do not provide your reasoning and just explicitly assign the appropriate label."},
                
                {"role": "user", "content": prompt}
            ],
            "stream": args.stream,
            "chat_template": chat_template
        }

        resp = requests.post(api_url, json=data)
        resp_json = resp.json()

        if "choices" in resp_json:
            print("-" * 50)
            print("PROMPT:\n", prompt)
            print("RESPONSE:\n", resp_json["choices"][0]["message"]["content"])
            print("-" * 50)
        else:
            print("Error from vLLM server:", resp_json)

if __name__ == "__main__":
    args = parse_args()
    main(args)


