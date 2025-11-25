"""
Agent that specializes in pricing strategies
it relies on model fine tuned
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
 deployed on modal
 https://colab.research.google.com/drive/1Vrp9hS8CDVPvpAMBido7Ws6C-EqJqo6C#scrollTo=VdB4LL6FIxgk

"""

import modal
from .agent import Agent


class SpecialistAgent(Agent):
    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        """Initialize the SpecialPricer agent and set up the modal stub."""
        self.log("Initializing SpecialPricer agent...")
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")

        self.pricer = Pricer()
        self.log("SpecialPricer agent Ready.")

    def price(self, description: str):  # -> float:
        """Get a price for the given description using the pricer service."""
        self.log(f"Getting price for description: {description}")
        result = self.pricer.price.remote(description)
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result
