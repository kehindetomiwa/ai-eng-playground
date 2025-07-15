from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
from openai import OpenAI
import ollama
import os


api_key = os.getenv("OPENAI_API_KEY")
MODEL_LLAMA = "llama3.2"
MODEL_GPT = "gpt-4o-mini"

load_dotenv(override=True)
openai = OpenAI()

question = """
Please explain what this code does and why:
yield from {book.get("autor") for book if book.get("author")}
"""

system_prompt = """ 
You are a helpful technical tutor who answers questions about python code, software engineering, data science and LLMS
"""

user_prompt = (
    """ Please give a detailed explanation of the following question:  """ + question
)


messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

stream = openai.chat.completions.create(model=MODEL_GPT, messages=messages, stream=True)

response = ""
display_handle = display(Markdown(""), display_id=True)
for chunk in stream:
    response += chunk.choices[0].delta.content or ""
    response = response.replace("```", "").replace("markdown", "")
    update_display(Markdown(response), display_id=display_handle.display_id)


response = ollama.chat(model=MODEL_LLAMA, messages=messages)
reply = response["message"]["content"]
display(Markdown(reply))
