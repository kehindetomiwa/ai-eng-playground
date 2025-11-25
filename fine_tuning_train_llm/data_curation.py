# imports

from items import Item
import os
import random
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import numpy as np
import pickle

from loaders import ItemLoader
from items import Item

"""
Download file frome Hugging Face -> preprocess -> then export to pickel file
Alternatvely, download pickel file from here:
https://drive.google.com/drive/folders/1f_IZGybvs9o0J5sb3xmtTEQB3BXllzrW
"""

path_env = "/Users/kehindetomiwa/Documents/Certifications/llm_udemy_ligency_team/ai-eng-playground/.env"


load_dotenv(path_env, override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-if-not-using-env")
os.environ["ANTHROPIC_API_KEY"] = os.getenv(
    "ANTHROPIC_API_KEY", "your-key-if-not-using-env"
)
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "your-key-if-not-using-env")


hf_token = os.environ["HF_TOKEN"]
login(hf_token, add_to_git_credential=True)


"""
The dataset is here:
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

And the folder with all the product datasets is here:
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/meta_categories

"""


def category_counter():
    category_counts = Counter()
    for item in items:
        category_counts[item.category] += 1

    categories = category_counts.keys()
    counts = [category_counts[category] for category in categories]

    # Bar chart by category
    plt.figure(figsize=(15, 6))
    plt.bar(categories, counts, color="goldenrod")
    plt.title("How many in each category")
    plt.xlabel("Categories")
    plt.ylabel("Count")

    plt.xticks(rotation=30, ha="right")

    # Add value labels on top of each bar
    for i, v in enumerate(counts):
        plt.text(i, v, f"{v:,}", ha="center", va="bottom")

    # Display the chart
    plt.savefig("category_counter.png")


def report(item):
    prompt = item.prompt
    tokens = Item.tokenizer.encode(item.prompt)
    print(prompt)
    print(tokens[-10:])
    print(Item.tokenizer.batch_decode(tokens[-10:]))


if __name__ == "__main__":
    dataset_names = [
        # "Automotive",
        # "Electronics",
        # "Office_Products",
        # "Tools_and_Home_Improvement",
        # "Cell_Phones_and_Accessories",
        # "Toys_and_Games",
        "Appliances",
        # "Musical_Instruments",
    ]
    items = []
    for dataset_name in dataset_names:
        loader = ItemLoader(dataset_name)
        items.extend(loader.load())

    tokens = [item.token_count for item in items]
    plt.figure(figsize=(15, 6))
    plt.title(
        f"Token counts: Avg {sum(tokens) / len(tokens):,.1f} and highest {max(tokens):,}\n"
    )
    plt.xlabel("Length (tokens)")
    plt.ylabel("Count")
    plt.hist(tokens, rwidth=0.7, color="skyblue", bins=range(0, 300, 10))
    # complete the code to save the figure to a file called token_counts_histogram.png
    plt.savefig("token_counts_histogram.png")

    sample = items
    report(sample[50])

    random.seed(42)
    random.shuffle(sample)
    train = sample[:25_000]
    test = sample[25_000:27_000]
    print(
        f"Divided into a training set of {len(train):,} items and test set of {len(test):,} items"
    )

    print(train[0].prompt)
    print(test[0].test_prompt())

    train_prompts = [item.prompt for item in train]
    train_prices = [item.price for item in train]
    test_prompts = [item.test_prompt() for item in test]
    test_prices = [item.price for item in test]

    # Create a Dataset from the lists

    train_dataset = Dataset.from_dict({"text": train_prompts, "price": train_prices})
    test_dataset = Dataset.from_dict({"text": test_prompts, "price": test_prices})
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    DATASET_NAME = "ktomiwacloudai/lite-data"
    dataset.push_to_hub(DATASET_NAME, private=True)

    # Let's pickle the training and test dataset so we don't have to execute all this code next time!

    with open("data/train_lite.pkl", "wb") as file:
        pickle.dump(train, file)

    with open("data/test_lite.pkl", "wb") as file:
        pickle.dump(test, file)
