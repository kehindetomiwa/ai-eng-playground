import os
import glob
from dotenv import load_dotenv
import gradio as gr
from multipart import file_path
from openai import OpenAI

MODEL = "gpt-4o-mini"

load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-if-not-using-env")
openai = OpenAI()


context = {}


def build_context_from_file(file_path="knowledge-base/employees/*", sep=" "):
    files = glob.glob(file_path)

    for file in files:
        name = file.split(sep)[-1][:-3]
        doc = ""
        with open(file, "r", encoding="utf-8") as f:
            doc = f.read()
        context[name] = doc


build_context_from_file()
build_context_from_file(file_path="knowledge-base/products/*", sep=os.sep)


system_message = "You are an expert in answering accurate questions about Insurellm, the Insurance Tech company. Give brief, accurate answers. If you don't know the answer, say so. Do not make anything up if you haven't been provided with relevant context."


def get_relevant_context(message):
    relevant_context = []
    for context_title, context_details in context.items():
        if context_title.lower() in message.lower():
            relevant_context.append(context_details)
    return relevant_context


def add_context(message):
    relevant_context = get_relevant_context(message)
    if relevant_context:
        message += "\n\nThe following additional context might be relevant in answering this question:\n\n"
        for relevant in relevant_context:
            message += relevant + "\n\n"
    return message


def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history
    message = add_context(message)
    messages.append({"role": "user", "content": message})

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response


if __name__ == "__main__":

    print(context.keys())
    view = gr.ChatInterface(chat, type="messages").launch()
