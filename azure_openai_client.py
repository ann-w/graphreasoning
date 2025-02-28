import os
import logging
import time
from openai import AzureOpenAI
from ratelimit import limits, sleep_and_retry

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AzureOpenAIClient:
    """Handles text generation using Azure OpenAI."""

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.max_retries = 5
        self.backoff_factor = 1

    @sleep_and_retry
    @limits(calls=20, period=60)  # Adjust the rate limit as needed
    def generate_response(self, messages: list) -> str:
        """Generate response with rate limiting and retries."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                    messages=messages,
                    max_tokens=512,
                    n=1,
                    stop=None,
                    temperature=0.7,
                )
                print(response.choices[0].message.content)
                return response.choices[0].message.content
            except Exception as e:
                wait_time = self.backoff_factor * (2**attempt)
                logger.warning(
                    f"Retry {attempt + 1}/{self.max_retries} after {wait_time}s: {str(e)}"
                )
                time.sleep(wait_time)
                if attempt == self.max_retries - 1:
                    raise

    def generate_completion(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.7
    ) -> str:
        """Generate a completion for a given prompt."""
        messages = [{"role": "user", "content": prompt}]
        return self.generate_response(messages)
