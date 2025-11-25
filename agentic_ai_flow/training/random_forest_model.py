# imports

import os
from dotenv import load_dotenv
from huggingface_hub import login
import chromadb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging

from agent_utils.random_forest_pricer import random_forest_model
from config_setting import ConfigSetting

config = ConfigSetting()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-if-not-using-env")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "your-key-if-not-using-env")

QUESTION = "How much does this cost to the nearest dollar?\n\n"
# TODO: remove hardcoded paths
DB =   config.chromadb_path
collection_name = config.collection_name

random_forest_model_path = config.random_forest_model_path


hf_token = os.environ["HF_TOKEN"]
login(hf_token, add_to_git_credential=True)

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # adds project root
# from items import Item


def main():
    logger.info("Starting data gathering for ML training...")

    client = chromadb.PersistentClient(path=DB)
    collection = client.get_or_create_collection(collection_name)

    result = collection.get(include=["embeddings", "documents", "metadatas"])
    vectors = np.array(result["embeddings"])
    documents = result["documents"]
    prices = [metadata["price"] for metadata in result["metadatas"]]
    logger.info("finished data gathering...\n")

    logger.info("Starting Random Forest Training...\n")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(vectors, prices)

    logger.info("Random Forest Training completed.n")
    logger.info("Saving Random Forest model to db/random_forest_model.pkl...\n")
    joblib.dump(rf_model, random_forest_model_path)
    logger.info("Finished training...")


if __name__ == "__main__":
    main()
