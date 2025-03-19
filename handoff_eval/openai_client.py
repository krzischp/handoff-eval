import asyncio
import os

import openai

client = openai.AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Default OpenAI model for completions
# DEFAULT_MODEL = "gpt-4-turbo"
DEFAULT_MODEL = "gpt-3.5-turbo"

# help stabilize API requests and avoid OpenAI’s retries
semaphore = asyncio.Semaphore(500)  # Limit to 500 concurrent API calls


async def call_openai_with_limit(model, messages, temperature):
    async with semaphore:
        return await client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
