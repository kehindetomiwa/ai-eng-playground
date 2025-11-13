import os
import re
import math
import json
import random
from dotenv import load_dotenv
from huggingface_hub import login
from items import Item
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import Counter
from openai import OpenAI
from anthropic import Anthropic

from testing import Tester

path_env = "/Users/kehindetomiwa/Documents/Certifications/llm_udemy_ligency_team/ai-eng-playground/.env"
load_dotenv(path_env, override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-if-not-using-env")
os.environ["ANTHROPIC_API_KEY"] = os.getenv(
    "ANTHROPIC_API_KEY", "your-key-if-not-using-env"
)
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "your-key-if-not-using-env")


openai = OpenAI()

with open("data/train_lite.pkl", "rb") as file:
    train = pickle.load(file)

with open("data/test_lite.pkl", "rb") as file:
    test = pickle.load(file)

fine_tune_train = train[:500]
fine_tune_validation = train[500:550]


def messages_for(item):
    system_message = (
        "You estimate prices of items. Reply only with the price, no explanation"
    )
    user_prompt = (
        item.test_prompt()
        .replace(" to the nearest dollar", "")
        .replace("\n\nPrice is $", "")
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": f"Price is ${item.price:.2f}"},
    ]


def make_jsonl(items):
    result = ""
    for item in items:
        messages = messages_for(item)
        messages_str = json.dumps(messages)
        result += '{"messages": ' + messages_str + "}\n"
    return result.strip()


def write_jsonl(items, filename):
    with open(filename, "w") as f:
        jsonl = make_jsonl(items)
        f.write(jsonl)


write_jsonl(fine_tune_train, "fine_tune_train.jsonl")
write_jsonl(fine_tune_validation, "fine_tune_validation.jsonl")

with open("fine_tune_train.jsonl", "rb") as f:
    train_file = openai.files.create(file=f, purpose="fine-tune")

with open("fine_tune_validation.jsonl", "rb") as f:
    validation_file = openai.files.create(file=f, purpose="fine-tune")

wandb_integration = {"type": "wandb", "wandb": {"project": "gpt-pricer"}}


openai.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=validation_file.id,
    model="gpt-4o-mini-2024-07-18",
    seed=42,
    hyperparameters={"n_epochs": 1},
    integrations=[wandb_integration],
    suffix="pricer",
)

openai.fine_tuning.jobs.list(limit=1)
job_id = openai.fine_tuning.jobs.list(limit=1).data[0].id

openai.fine_tuning.jobs.retrieve(job_id)
openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10).data

fine_tuned_model_name = openai.fine_tuning.jobs.retrieve(job_id).fine_tuned_model


# The prompt


def messages_for(item):
    system_message = (
        "You estimate prices of items. Reply only with the price, no explanation"
    )
    user_prompt = (
        item.test_prompt()
        .replace(" to the nearest dollar", "")
        .replace("\n\nPrice is $", "")
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Price is $"},
    ]


messages_for(test[0])


def get_price(s):
    s = s.replace("$", "").replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0


def gpt_fine_tuned(item):
    response = openai.chat.completions.create(
        model=fine_tuned_model_name, messages=messages_for(item), seed=42, max_tokens=7
    )
    reply = response.choices[0].message.content
    return get_price(reply)


Tester.test(gpt_fine_tuned, test)
