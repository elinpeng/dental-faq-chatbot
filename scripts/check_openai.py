"""Check that the OpenAI API key in .env works and that the configured chat model answers."""
from dotenv import load_dotenv
from openai import OpenAI

from src.config import LLM_MODEL, ROOT_DIR

load_dotenv(ROOT_DIR / ".env")
client = OpenAI()
ids = sorted(m.id for m in client.models.list().data)
print(f"API key works; {len(ids)} models available")
print("Small models:", [i for i in ids if "mini" in i or "nano" in i])
reply = client.chat.completions.create(
    model=LLM_MODEL, messages=[{"role": "user", "content": "Reply with the word OK."}]
)
print(f"{LLM_MODEL} reply: {reply.choices[0].message.content}")
