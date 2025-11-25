import os
import logging
import re

from dotenv import load_dotenv
from openai import OpenAI
import anthropic


logging.basicConfig(
    level=logging.WARNING, format="%(levelname)s:%(name)s:%(funcName)s:%(message)s"
)

# check if API keys are in .env
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

assert openai_api_key, "OpenAI API key is missing"
assert anthropic_api_key, "Anthropic API key is missing"


class Model:
    """One class for different API's.

    This implementation allows the use of the OpenAI and Anthropic API. Other endpoints,
    such as Ollama, can be used as well, as long as they are used via the OpenAI
    Python library.

    """

    def __init__(self, api=None, model_name="mock"):
        """
        Args:
            api: Can be an OpenAI or anthropic.Anthropic object or None to make a mock run.
            model_name (str): Identifies the model used via the API.

        """
        self.api = api
        self.name = model_name
        if type(self.api) not in {OpenAI, anthropic.Anthropic} and self.name not in {
            "mock",
            "",
        }:
            logging.warning(f"Unknown API '{self.api}'. Using mock.")

    def complete(self, messages, system=""):
        """Make API call."""
        completion = ""
        if isinstance(self.api, OpenAI):
            completion = self.api.chat.completions.create(
                model=self.name,
                messages=[{"role": "system", "content": system}] + messages,
                max_tokens=300,
            )
            completion = completion.choices[0].message.content

        elif isinstance(self.api, anthropic.Anthropic):
            completion = self.api.messages.create(
                model=self.name, system=system, messages=messages, max_tokens=300
            )
            completion = completion.content[0].text

        else:
            completion = "Mock answer."

        return self.parse_answer(completion)

    def parse_answer(self, answer):
        # Remove prefix 'Name:' from answer if present.
        regex = r"(?P<name>\w+): (?P<content>.*)"
        match = re.match(regex, answer, re.DOTALL)
        if match:
            logging.info(f"{self.name} generated {match.group('name')}")
            return match.group("content")
        return answer


class Participant:
    """Represents one participant in a conversation."""

    def __init__(self, name, model=Model(), system_prompt="", initial_message=""):
        """
        Args:
            model (Model): The model that is called to get participant's answer.
            name (str): Used to assign answers to different participants. Is inserted in the
                messages list, so the model knows who's spoken. Is also
                displayed in the output.
            system_prompt (str): The system prompt overgiven to the model backend.
            initial_message (str): An optional conversation start.
        """
        self.model = model
        self.name = name
        self.role = system_prompt
        self.initial_msg = initial_message
        self.messages = []  # keeps conversation history
        self.last_msg = ""

    def speak(self):
        if self.initial_msg:
            self.last_msg = self.initial_msg
            self.initial_msg = ""
        else:
            self.last_msg = self.model.complete(self.messages, self.role)
        self.update_messages(role="assistant", content=self.last_msg)
        return self.last_msg

    def listen(self, message: str, speaker_name: str):
        # Insert the speaker name, so the model can distinguish them
        self.update_messages(role="user", content=f"{speaker_name}: {message}")

    def update_messages(self, role, content):
        self.messages.append({"role": role, "content": content})


class ThreeWayChat:
    """Make three Participants communicate."""

    def __init__(self, participants, n_turns=4):
        """
        Args:
            participants (tuple[Participant]): Three objects. The order determines the speaking order.
            n_turns (int): Number of turns per participant, incl. Participant.initial_message.

        """
        self.n_turns = n_turns
        self.p1, self.p2, self.p3 = participants
        if (
            len(
                {
                    bool(self.p1.initial_msg),
                    bool(self.p2.initial_msg),
                    bool(self.p3.initial_msg),
                }
            )
            != 1
        ):
            logging.warning(
                "At least one Participant has gotten a value for initial_message while another hasn't."
            )
        if len({self.p1.name, self.p2.name, self.p3.name}) != 3:
            raise ValueError(
                f"Some Participants have the same name. "
                f"Please use unique names."
                f"\nNames you've given: {self.p1.name}, {self.p2.name} and {self.p3.name}. "
            )

    def start(self, n_turns=None):
        """Start a conversation with n_turns rounds.

        Args:
            n_turns (int): If None, self.n_turns is used.

        """
        for i in range(n_turns or self.n_turns):
            # Make each participant speak and display their answers
            self.make_display_turn(self.p1, self.p2, self.p3)
            self.make_display_turn(self.p2, self.p1, self.p3)
            self.make_display_turn(self.p3, self.p2, self.p1)

    def make_display_turn(self, speaker, *listeners):
        self.speaker_to_listeners(speaker, *listeners)
        self.display_last_utterance(speaker)

    def speaker_to_listeners(self, speaker, *listeners):
        """Get answer from speaker and update conversation histories."""
        speaker_text = speaker.speak()
        for listener in listeners:
            listener.listen(speaker_text, speaker.name)

    def display_last_utterance(self, speaker):
        print(
            "{} ({}):\n{}\n".format(
                speaker.name.upper(), speaker.model.name, speaker.last_msg
            )
        )


name1 = "Austin"
name2 = "Jonas"
name3 = "Tim"

general_system = (
    "\n\nYou've entered a chatroom with two other participants. "
    'Their names are "{}" and "{}". Your name is "{}".'
    "\nGenerate a maximum of 100 words per turn."
)

system1 = (
    "You are very argumentative; "
    "You always find something to discuss. "
    "When someone says their opinion, you often disagree. "
    "You enjoy swimming against the tide and mocking mainstream opinions."
    + general_system.format(name3, name2, name1)
)

system2 = (
    "You have a very conservative and clear opinion on most things. "
    "You feel safest in your familiar surroundings. You are very reluctant to try out new things. "
    "In discourses you are stubborn and want to convince others from your gridlocked beliefs."
    + general_system.format(name1, name3, name2)
)

system3 = (
    "You are very humorous and like to be ironic. Sometimes you tell silly jokes. "
    "You like variation; If a discussion about a topic takes too long, you start a new topic."
    + general_system.format(name1, name2, name3)
)

openai_api = OpenAI()
claude_api = anthropic.Anthropic()
# ollama could be used like this:
# ollama_api = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

claude_model_str = "claude-3-haiku-20240307"
gpt_model_str = "gpt-4o-mini"
# llama_model_str = "llama3.2"


# Create Model objects
gpt_model = Model(openai_api, gpt_model_str)
claude_model = Model(claude_api, claude_model_str)

# Create three Participants
p1 = Participant(
    name=name1, model=gpt_model, system_prompt=system1, initial_message="Hello there"
)
p2 = Participant(
    name=name2,
    model=claude_model,
    system_prompt=system2,
    initial_message="Good evening.",
)
p3 = Participant(
    name=name3, model=gpt_model, system_prompt=system3, initial_message="Hey guys"
)

# To make a mock run without API calls:
# p1 = Participant(name=name1, system_prompt=system1, initial_message="Hello there")
# p2 = Participant(name=name2, system_prompt=system2, initial_message="Good evening.")
# p3 = Participant(name=name3, system_prompt=system3, initial_message="Hey guys")

# Create Chat
chat = ThreeWayChat((p1, p2, p3))
chat.start()  # starts a chat with 4 rounds
# chat.start(2) # 2 rounds
