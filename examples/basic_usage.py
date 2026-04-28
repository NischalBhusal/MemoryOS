"""Basic MemoryOS usage — no Ollama required."""

from memoryos import Config, Memory


def main() -> None:
    cfg = Config(db_path="basic_usage.db", log_level="INFO")

    with Memory(config=cfg) as mem:
        mem.clear()

        print("\n[+] Storing memories…")
        mem.add("User prefers dark mode")
        mem.add("User is learning Python and building AI apps")
        mem.add("User's favourite food is sushi")

        print(f"\n    Total stored: {mem.count()}")

        print("\n[?] Searching: 'what does the user like?'")
        for r in mem.search("what does the user like?", top_k=2):
            print(f"    {r}")

        print("\n[!] Forgetting about dark mode…")
        removed = mem.forget("dark mode", threshold=0.3)
        print(f"    Removed: {removed}")

        print("\n[?] Searching again for 'dark mode'")
        results = mem.search("dark mode", top_k=1, min_score=0.5)
        if not results:
            print("    ✓ Memory successfully forgotten.")
        else:
            print(f"    Still found: {results[0]}")

        print(f"\n    Final count: {mem.count()}")


if __name__ == "__main__":
    main()
