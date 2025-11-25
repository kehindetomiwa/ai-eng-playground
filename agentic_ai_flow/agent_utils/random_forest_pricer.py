# imports

"""A Random Forest based agent to estimate product prices"""
from sentence_transformers import SentenceTransformer
import joblib
from .agent import Agent
from config_setting import ConfigSetting
config = ConfigSetting()

random_forest_model = config.random_forest_model_path


class RandomForestAgent(Agent):

    name = "Random Forest Agent"
    color = Agent.MAGENTA

    def __init__(self):
        """
        Initialize this object by loading in the saved model weights
        and the SentenceTransformer vector encoding model
        """
        self.log("Random Forest Agent is initializing")
        self.vectorizer = SentenceTransformer(config.victorization_model_name)
        self.model = joblib.load(random_forest_model)
        self.log("Random Forest Agent is ready")

    def price(self, description: str) -> float:
        """
        Use a Random Forest model to estimate the price of the described item
        :param description: the product to be estimated
        :return: the price as a float
        """
        self.log("Random Forest Agent is starting a prediction")
        vector = self.vectorizer.encode([description])
        result = max(0, self.model.predict(vector)[0])  # inference step
        self.log(f"Random Forest Agent completed - predicting ${result:.2f}")
        return result
