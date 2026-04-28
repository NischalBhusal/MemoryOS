"""
Ollama + MemoryOS integration demo.

Shows summarize() and forget() with AI-powered compression.
Requires Ollama running: ollama serve && ollama pull llama3
"""

from memoryos import Config, Memory
from memoryos.exceptions import OllamaUnavailableError


def main() -> None:
    cfg = Config(db_path="ollama_demo.db", llm_model="llama3")

    with Memory(config=cfg) as mem:
        mem.clear()

        print("\n[+] Loading memories…")
        facts = [
            "Meeting with John tomorrow at 10 AM about the Apollo project.",
            "Need to buy: milk, eggs, and coffee.",
            "Secret project codename is 'Apollo'.",
            "User enjoys hiking and outdoor activities.",
            "User is vegetarian and allergic to nuts.",
        ]
        for f in facts:
            mem.add(f)

        print(f"    Stored {mem.count()} memories.\n")

        try:
            print("[AI] Generating summary of ALL memories…\n")
            summary = mem.summarize()
            print("--- Summary ---")
            print(summary)
            print("---------------\n")

            print("[!] Forgetting the project codename…")
            removed = mem.forget("secret project codename Apollo")
            print(f"    Removed: {removed}\n")

            print("[AI] Re-generating summary after forget…\n")
            summary2 = mem.summarize()
            print("--- Updated Summary ---")
            print(summary2)
            print("-----------------------\n")

        except OllamaUnavailableError as e:
            print(f"\n[MemoryOS] {e}")
            print("Tip: run `ollama serve` and `ollama pull llama3` first.")


if __name__ == "__main__":
    main()
