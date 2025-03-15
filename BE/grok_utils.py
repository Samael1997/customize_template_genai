from typing import Dict, List
from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI


class GroqClient:
    
    def __init__(self,):
        """
        Initializes the Groq client using the OpenAI API format.
        
        :param api_key: Your Groq API key.
        :param model: The model name to use (default: "llama3-8b-8192").
        """
        

        self.client = OpenAI(api_key=os.environ['GROQ_API_KEY'], base_url=os.environ['GROQ_BASE_URL'])
        self.model = os.environ['GROQ_MODEL']

    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 500):
        """
        Sends a chat completion request to the Groq model.
        
        :param messages: A list of messages in OpenAI format [{'role': 'user', 'content': 'Hello'}].
        :param temperature: Controls randomness (0.0 for deterministic output).
        :param max_tokens: Maximum number of tokens to generate.
        :return: The model's response as a string.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

