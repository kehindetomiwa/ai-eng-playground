from .agent import Agent
# from items import Item


class Pricer(Agent):
    def __init__(self, init_param=None):
        name = "Frontier Agent"
        color = Agent.BLUE
        """Initialize the Pricer agent."""
        self.init_param = init_param
        print("Pricer agent initialized.")

    def price(self, description: str):
        """A mock pricing function that returns a price based on the length of the description."""
        print(
            f"Calculating price for description: {description} initialized with {self.init_param}"
        )
        self.log(f" PPP for description {self.name} {self.color}")
