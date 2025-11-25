# imports

import os
from cProfile import label

import requests
from bs4 import BeautifulSoup
from typing import List
from dotenv import load_dotenv
from grpc.framework.interfaces.base.utilities import completion
from openai import OpenAI
import google.generativeai
import anthropic


import gradio as gr

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")


openai = OpenAI()

claude = anthropic.Anthropic()

google.generativeai.configure()


system_message = "You are a helpful assistant"


def message_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return completion.choices[0].message.content


def stream_message_gpt(prompt: str) -> List[str]:
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    stream = openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages, stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result


def stream_message_claude(prompt):
    result = claude.messages.stream(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.7,
        system=system_message,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response = ""
    with result as stream:
        for text in stream.text_stream:
            response += text or ""
            yield response


def stream_model(prompt, model):
    if model == "GPT":
        result = stream_message_gpt(prompt)
    elif model == "Claude":
        result = stream_message_claude(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result


def shout(prompt: str) -> str:
    """Convert the input prompt to uppercase."""
    print(f"Shout has been called with input {prompt}")
    return prompt.upper()


def simple_gradio_ui(fxn, fxt_input):
    gr.Interface(
        fn=fxn, inputs="textbox", outputs="textbox", flagging_mode="never"
    ).launch(share=True)


def interface_with_textbox(fxn, fxt_input):
    """
    Create a Gradio interface with a textbox input and output.
    """
    gr.Interface(
        fn=fxn,
        inputs=gr.Textbox(label="Input", lines=6),
        outputs=gr.Markdown(
            label="Response"
        ),  # outputs=gr.Textbox(label="Output", lines=10),
        title="Use LLM via Gradio, CC KENNY",
        description="Use Kenny GPT to process your input.",
        flagging_mode="never",
    ).launch(inbrowser=True, share=True)


def interface_llm_selector(fxn, fxt_input):
    """
    Create a Gradio interface with a model selector and a textbox input.
    """
    view = gr.Interface(
        fn=stream_model,
        inputs=[
            gr.Textbox(label="Your message:"),
            gr.Dropdown(["GPT", "Claude"], label="Select model", value="GPT"),
        ],
        outputs=[gr.Markdown(label="Response:")],
        flagging_mode="never",
    )
    view.launch()


# A class to represent a Webpage


class Website:
    url: str
    title: str
    text: str

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"


def stream_brochure(company_name, url, model):
    prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
    prompt += Website(url).get_contents()
    if model == "GPT":
        result = stream_message_gpt(prompt)
    elif model == "Claude":
        result = stream_message_claude(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result


def interface_create_brochure_from_url():
    view = gr.Interface(
        fn=stream_brochure,
        inputs=[
            gr.Textbox(label="Company name:"),
            gr.Textbox(label="Landing page URL including http:// or https://"),
            gr.Dropdown(["GPT", "Claude"], label="Select model"),
        ],
        outputs=[gr.Markdown(label="Brochure:")],
        flagging_mode="never",
    )
    view.launch(inbrowser=True)


if __name__ == "__main__":
    # simple_gradio_ui(shout, "Enter your prompt here")

    # int_with_textbox(message_gpt, "Enter your prompt here")

    # interface_with_textbox(stream_message_gpt(), "Enter your prompt here")

    # interface_llm_selector(stream_model, "Enter your prompt here")

    interface_create_brochure_from_url()
