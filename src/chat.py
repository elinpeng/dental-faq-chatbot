"""Terminal chat with the dental FAQ chatbot.

Run from the project root:
    .venv/bin/python -m src.chat
"""
import os
import sys

from src.rag_chain import DentalRAG


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is missing. Add it to the .env file in the project root.")

    print("Loading the dental FAQ chatbot...")
    rag = DentalRAG()
    print("Ask a dental question (type 'quit' to exit).\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question:
            # Only the answer text is shown; source metadata stays on the result object.
            print(f"\nBot: {rag.ask(question).answer}\n")


if __name__ == "__main__":
    main()
