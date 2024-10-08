import openai
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIWrapper:
    def __init__(self, openai_key):
        self.openai_key = openai_key
        self.client = OpenAI(api_key=self.openai_key)

    def chat_completion(self, prompt):
        response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.8,
            )
        message = response.choices[0].message.content

        return message
 