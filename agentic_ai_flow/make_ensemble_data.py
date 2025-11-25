"""
module calls agents in agent_utils to make ensemble data for pricing agents:
    * frontier pricer agent: frontier_pricer_agent.py
    * special pricer agent: special_pricer.py
    * random forest pricer agent: random_forest_pricer.py
"""

import os
from tqdm import tqdm
from dotenv import load_dotenv
import pickle
import chromadb
import pandas as pd
import numpy as np
from agent_utils import frontier_pricer_agent, special_pricer, random_forest_pricer
from agent_utils import pricer_pp
from config_setting import ConfigSetting

config = ConfigSetting()


load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-if-not-using-env")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "your-key-if-not-using-env")

os.environ["MODAL_TOKEN_ID"] = os.getenv("MODAL_TOKEN_ID", "your-key-if-not-using-env")
os.environ["MODAL_TOKEN_SECRET"] = os.getenv(
    "MODAL_TOKEN_SECRET", "your-key-if-not-using-env"
)


DB = config.chromadb_path
collection_name = config.collection_name

client = chromadb.PersistentClient(path=DB)
collection = client.get_or_create_collection("products")

result = collection.get(include=["embeddings", "documents", "metadatas"])
vectors = np.array(result["embeddings"])
documents = result["documents"]
prices = [metadata["price"] for metadata in result["metadatas"]]

# TODO: remove hardcoded paths
test_data = config.test_data_path
output_data = config.ensemble_data_path


def description(item):
    return item.prompt.split("to the nearest dollar?\n\n")[1].split("\n\nPrice is $")[0]


if __name__ == "__main__":
    """
    When run as a script, create the ensemble data
    """
    f_obj = pricer_pp.Pricer(init_param="Initialization Parameter")
    for x in range(5):
        f_obj.price("Sample product description")

    with open(test_data, "rb") as file:
        test = pickle.load(file)

    specialist = special_pricer.SpecialistAgent()
    frontier = frontier_pricer_agent.FrontierAgent(collection)
    random_forest = random_forest_pricer.RandomForestAgent()

    specialists = []
    frontiers = []
    random_forests = []
    prices = []
    for item in tqdm(test[1000:1250]):
        text = description(item)

        specialists.append(specialist.price(description=text))
        frontiers.append(frontier.price(description=text))
        random_forests.append(random_forest.price(description=text))
        prices.append(item.price)

    mins = [min(s, f, r) for s, f, r in zip(specialists, frontiers, random_forests)]
    maxes = [max(s, f, r) for s, f, r in zip(specialists, frontiers, random_forests)]

    data = pd.DataFrame(
        {
            "Specialist": specialists,
            "Frontier": frontiers,
            "RandomForest": random_forests,
            "Min": mins,
            "Max": maxes,
            "Price": prices,
        }
    )

    data.to_pickle(output_data)
