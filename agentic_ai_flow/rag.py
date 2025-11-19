""" "

RAG (Retrieval-Augmented Generation) process for product data.
vectorstore using ChromaDB and Sentence Transformers.
This script performs the following steps:
1. Load product data from a dataset *data/*.pkl.
2. Preprocess and clean the data.
3. Generate embeddings using a Sentence Transformer model.
4. Store embeddings in a ChromaDB vectorstore.
5. (Optional) Visualize embeddings using t-SNE and Plotly.

"""

import os
import re
import math
import json
from tqdm import tqdm
import random
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import chromadb
from sklearn.manifold import TSNE
import plotly.graph_objects as go


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-if-not-using-env")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "your-key-if-not-using-env")

DB = "db/products_vectorstore"
collection_name = "products"

# Model for generating embeddings   maps sentences & paragraphs to a 384 dimensional dense
# vector space and is ideal for tasks like semantic search.
victorization_model_name = "sentence-transformers/all-MiniLM-L6-v2"


hf_token = os.environ["HF_TOKEN"]
login(hf_token, add_to_git_credential=True)

from items import Item


def description(item):
    text = item.prompt.replace("How much does this cost to the nearest dollar?\n\n", "")
    return text.split("\n\nPrice is $")[0]


def main():
    logger.info("Starting RAG process...")
    # # Load dataset
    with open("data/train_lite.pkl", "rb") as f:
        train = pickle.load(f)
    logger.info(f"Loaded dataset with {len(train)} records.")
    logger.info(f"Preprocessing first item  {train[0].prompt}...")

    client = chromadb.PersistentClient(path=DB)

    # check if collection exit and delete then create new
    if collection_name in [col.name for col in client.list_collections()]:
        client.delete_collection(collection_name)
    collection = client.create_collection(name=collection_name)
    model = SentenceTransformer(victorization_model_name)

    # Process and add items to vectorstore

    NUMBER_OF_ITEMS = len(train)

    for i in tqdm(range(0, NUMBER_OF_ITEMS, 1000)):
        documents = [description(item) for item in train[i : i + 1000]]

        embeddings = model.encode(documents).astype(float).tolist()
        metadatas = [
            {"categoty": item.category, "price": item.price}
            for item in train[i : i + 1000]
        ]
        ids = [f"doc_{j}" for j in range(i, i + len(documents))]
        collection.add(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )

    logger.info(f"Finished adding {NUMBER_OF_ITEMS} items to vectorstore: {DB}")


if __name__ == "__main__":
    main()
