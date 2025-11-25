# imports

import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# Load environment variables in a file called .env
# Print the key prefixes to help with any debugging

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")


# Initialize

openai = OpenAI()
MODEL = "gpt-4o-mini"


# Simpler than in my video - we can easily create this function that calls OpenAI
# It's now just 1 line of code to prepare the input to OpenAI!

# Student Octavio O. has pointed out that this isn't quite as straightforward for Claude -
# see the excellent contribution in community-contributions "Gradio_issue_with_Claude" that handles Claude.

system_message = "you are a Gospel Pastor, you have read the KJV version of the bible. Preach jesus to your user\
For example, if the user says hello I am looking for help, \
You could reply something like 'wonderful - Jesus loves to help you, His purpose to to give you eternal life'  make \
reference to bible passages. \
alway use He (with capital H) for Jesus pronoune."


def chat(message, history):
    messages = (
        [{"role": "system", "content": system_message}]
        + history
        + [{"role": "user", "content": message}]
    )

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response


if __name__ == "__main__":
    gr.ChatInterface(
        fn=chat,
        type="messages",
        title="KEHINDE LLM bot Prototype",
    ).launch(share=True)
