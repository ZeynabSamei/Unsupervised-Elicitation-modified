import asyncio
import json
import random
from collections import Counter
from copy import deepcopy
from datasets import load_dataset
from core.llm_api.llm import ModelAPI
from core.utils import setup_environment
from src.tools.path_utils import get_root_directory

# -----------------------
# System prompt for moral dataset
# -----------------------
SYSTEM_PROMPT = "You are an intelligent and objective assistant. Carefully read each statement and determine whether it is morally True (label 1) or False (label 0). Do not provide your reasoning and just explicitly assign the appropriate label."

# -----------------------
# Load data
# -----------------------
def load_data(args):
    if args.testbed == "moral_dataset":
        with open(get_root_directory() / "data/train_moral_dataset.json") as f:
            train = json.load(f)

        template = """Claim: {claim}
I think this claim is **** """

        for i in train:
            i['source'] = 'moral_dataset'
            i['consistency_key'] = 'A' if i['label'] else 'B'
            # add system prompt at the beginning
            i['prompt'] = SYSTEM_PROMPT + "\n" + template.format(claim=i['claim'])

    # sample a batch of batch_size
    fewshot_ids = random.sample(range(len(train)), args.batch_size)
    return train, fewshot_ids

# -----------------------
# Initialize labels
# -----------------------
def initialize(train, fewshot_ids, args):
    demonstrations = {}
    seed_ids = []
    random_init_labels = [1] * (args.num_seed // 2) + [0] * (args.num_seed // 2)
    random.shuffle(random_init_labels)

    for idx, i in enumerate(fewshot_ids):
        item = deepcopy(train[i])
        item["vanilla_label"] = item["label"]
        item["uid"] = idx
        if idx >= args.num_seed:
            item["label"] = None
            item["type"] = "predict"
        else:
            item["label"] = random_init_labels[idx]
            item["type"] = "seed"
            seed_ids.append(idx)
        demonstrations[idx] = item
    return demonstrations, seed_ids

# -----------------------
# Predict label using simple base model
# -----------------------
async def predict_label(model, example):
    response = await model(
        example["prompt"],
        logprobs=20,
        max_tokens=1
    )
    score = response[0]["score"]  # assuming model API returns this
    return int(score > 0)

# -----------------------
# Main
# -----------------------
def main(args):
    train, fewshot_ids = load_data(args)
    demonstrations, seed_ids = initialize(train, fewshot_ids, args)

    model_api = ModelAPI(anthropic_num_threads=20, openai_fraction_rate_limit=0.99)

    # predict for unlabeled items
    for idx, example in demonstrations.items():
        if example["label"] is None:
            new_label = asyncio.run(predict_label(model_api, example))
            example["label"] = new_label

    # Print results
    print("Label distribution:", Counter([i["label"] for i in demonstrations.values()]))

# -----------------------
# Args
# -----------------------
class Args:
    testbed = "moral_dataset"
    batch_size = 16
    num_seed = 4

if __name__ == "__main__":
    setup_environment(logger_level="error")
    args = Args()
    main(args)

