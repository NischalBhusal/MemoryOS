"""
Chatbot with persistent memory backed by MemoryOS + Ollama.

Start Ollama first:  ollama serve
Pull a model first:  ollama pull llama3
"""

from __future__ import annotations

import ollama

from memoryos import Config, Memory
from memoryos.exceptions import OllamaUnavailableError

SESSION_ID = "chat_001"


def build_system_prompt(context: str) -> str:
    return (
        "You are a helpful assistant with access to memory about the user.\n"
        "Use the following context from past interactions if relevant:\n"
        f"{context}\n"
        "Answer concisely and naturally."
    )


def main() -> None:
    cfg = Config(db_path="chatbot_memory.db", llm_model="llama3")

    with Memory(config=cfg) as mem:
        print("=" * 55)
        print(" AI Chatbot with Persistent MemoryOS Memory")
        print(f" Session: {SESSION_ID}  |  Total memories: {mem.count()}")
        print(" Type 'quit' to exit & auto-summarise the session.")
        print("=" * 55 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                break

            # Store user's turn
            mem.add(f"User: {user_input}", session_id=SESSION_ID)

            # Pull relevant context
            ctx_results = mem.search(user_input, top_k=3)
            context = "\n".join(f"- {r.text}" for r in ctx_results) or "(no prior context)"

            # Ask Ollama
            try:
                resp = ollama.chat(
                    model=cfg.llm_model,
                    messages=[
                        {"role": "system", "content": build_system_prompt(context)},
                        {"role": "user", "content": user_input},
                    ],
                )
                if hasattr(resp, "message"):
                    reply = resp.message.content
                else:
                    reply = resp["message"]["content"]

                print(f"Bot: {reply}\n")
                mem.add(f"Bot: {reply}", session_id=SESSION_ID)

            except OllamaUnavailableError as e:
                print(f"[MemoryOS] {e}\n")
                break
            except Exception as e:
                print(f"[Ollama error] {e}\n")
                break

        # Compress session into long-term memory
        print("\n--- Summarising session into long-term memory… ---")
        summary_id = mem.remember(session_id=SESSION_ID)
        if summary_id:
            print(f"✓ Session stored as memory id={summary_id}")
        print(f"Total memories now: {mem.count()}")


if __name__ == "__main__":
    main()
