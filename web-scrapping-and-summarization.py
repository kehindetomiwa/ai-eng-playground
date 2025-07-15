"""
Wrapper Class to scrape a webpage and summarize its content.
using requests and BeautifulSoup for web scraping.

This code uses the OpenAI API to summarize the content of a webpage.
model should be set to 'llama3'
current implemetation runs ollama via OpenAI API locally.
"""

import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# openai = OpenAI()
# model = os.getenv('MODEL', 'llama3')
MODEL = "llama3.2"


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}


class Website:

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)


system_prompt = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."


def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "\nThe contents of this website is as follows; \
please provide a short summary of this website in markdown. \
If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt


def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)},
    ]


from openai import OpenAI

ollama_via_openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def summarize(url):
    website = Website(url)
    response = ollama_via_openai.chat.completions.create(
        model=MODEL, messages=messages_for(website)
    )
    return response.choices[0].message.content


def display_summary(url):
    summary = summarize(url)
    display(Markdown(summary))


if __name__ == "__main__":
    display_summary("https://edwarddonner.com")
