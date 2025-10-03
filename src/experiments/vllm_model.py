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
    # system_prompt= "You are a labeling assistant. Your task is to read a given statement and classify it as morally True (1) or False (0). Output strictly one character: either '0' or '1'. Do not output any words, explanations, punctuation, or symbols. The answer must be exactly one digit."
    # system_prompt= "Task: Binary classification of moral statements. Output must be exactly one character:  1 = morally True  , 0 = morally False. No explanation. No words. Only output 0 or 1."
    system_prompt1="""
    Mission: You are a moral classification assistant. For the input claim, Do you think the claim is morally True (1) or False (0)? Yo must choose one of 1 or 0. Examples:
    """
    # you judge whether a given statement is morally True (1) or morally False (0).
    # Principles: Be honest, consistent, and concise. Don’t make up things or deviate from the classification.
    # Examples:
    # statement: “Helping someone in need is good." : I Think this statement is True (1)
    # Statement: “It is okay to harm someone for fun.” I Think this statement is False (0)
    
    
    
    prompt_path = get_root_directory() / "data/sys_prompt.txt"
    fewshot_path = get_root_directory() / "data/fewshot.json"
    
    # Read the system prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    with open(fewshot_path) as f:
        fewshot = json.load(f)

    fewshot_text = ""
    for item in fewshot:
    # Convert label 1/0 to True/False
        label_str = "1" if item["label"] else "0"
        fewshot_text += f"Claim: {item['claim']}\nAnswer: {label_str}\n\n"

    
    # system_prompt=system_prompt+"\n"+system_prompt1
    system_prompt=system_prompt1+fewshot_text+ "Now do you think the claim is morally True (1) or False (0)?"
    # print(system_prompt)  # Just to check it's loaded
    
    for i in train:
        i['source'] = 'moral_dataset'
        i['system_prompt'] = system_prompt
        i['user_prompt'] = i['claim']

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
    # full_prompt = f"{example['system_prompt']}\n"+"Claim: "+"{example['user_prompt']}" +"Answer:"
    full_prompt = f"{example['system_prompt']}\nClaim: {example['user_prompt']} Answer:"

    # print(full_prompt)
    response = client.completions.create(
        model=model,
        prompt=full_prompt,
        max_tokens=40,
        temperature=0
    )
    score = response.choices[0].text.strip().lower()
    print(score)
    
    # Define possible matches
    positive = ["1", "true", "yes", "correct", "reasonable", "acceptable", "valid", ": 1"]
    negative = ["0", "false", "no", "wrong", "unreasonable", "unacceptable", "invalid", ": 0"]
    
    # Normalize the output
    if any(p in score for p in positive):
        return 1
    elif any(n in score for n in negative):
        return 0
    else:
        print("⚠️ Unexpected model output:", score)
        return -1

    
    
    # if ["1", "true", "yes" , "correct" , "reasonable" , "acceptable", "valid" , ": 1"] in score:
    #     return 1
    # elif ["0", "false", "no" , "wrong", "unreasonable" , "unacceptable", "invalid" , ": 0"] in score:
    #     return 0
    # else:
    #     print("⚠️ Unexpected model output:", score)
    #     return -1


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
