# imports

import os
import json
from dotenv import load_dotenv
from langchain_core.messages.tool import tool_call
from openai import OpenAI
import gradio as gr

# Initialization

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

MODEL = "gpt-4o-mini"
openai = OpenAI()

# As an alternative, if you'd like to use Ollama instead of OpenAI
# Check that Ollama is running for you locally (see week1/day2 exercise) then uncomment these next 2 lines
# MODEL = "llama3.2"
# openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')


system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."


# This function looks rather simpler than the one from my video, because we're taking advantage of the latest Gradio updates

#
# def chat(message, history):
#     messages = (
#         [{"role": "system", "content": system_message}]
#         + history
#         + [{"role": "user", "content": message}]
#     )
#     response = openai.chat.completions.create(model=MODEL, messages=messages)
#     return response.choices[0].message.content
#
#
# gr.ChatInterface(fn=chat, type="messages").launch()

# ***** Tools *****

# Let's start by making a useful function

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}


def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")


# There's a particular dictionary structure that's required to describe our function:
price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False,
    },
}

weather_condition = {
    "london": "Rainy",
    "paris": "Sunny",
    "tokyo": "Cloudy",
    "berlin": "Windy",
}


def get_weather(destination_city):
    print(f"Tool get_weather called for {destination_city}")
    city = destination_city.lower()
    return weather_condition.get(city, "Unknown")


weather_function = {
    "name": "get_weather",
    "description": "Get the weather in the destination city. Call this whenever you need to know the weather, for example when a customer asks 'What's the weather like in this city?'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to know the weather for",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False,
    },
}

# And this is included in a list of tools:

tools = [
    {"type": "function", "function": price_function},
    {"type": "function", "function": weather_function},
]


# We have to write that function handle_tool_call:


def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    if tool_call.function.name == "get_ticket_price":
        arguments = json.loads(tool_call.function.arguments)
        city = arguments.get("destination_city")
        price = get_ticket_price(city)
        response = {
            "role": "tool",
            "content": json.dumps({"destination_city": city, "price": price}),
            "tool_call_id": tool_call.id,
        }
        return response, city
    if tool_call.function.name == "get_weather":
        arguments = json.loads(tool_call.function.arguments)
        city = arguments.get("destination_city")
        weather = get_weather(city)
        response = {
            "role": "tool",
            "content": json.dumps({"destination_city": city, "weather": weather}),
            "tool_call_id": tool_call.id,
        }
        return response, city
    else:
        raise ValueError(f"Unknown tool call: {tool_call.function.name}")


def chat(message, history):
    messages = (
        [{"role": "system", "content": system_message}]
        + history
        + [{"role": "user", "content": message}]
    )
    response = openai.chat.completions.create(
        model=MODEL, messages=messages, tools=tools
    )

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)

    return response.choices[0].message.content


if __name__ == "__main__":
    gr.ChatInterface(fn=chat, type="messages").launch()
