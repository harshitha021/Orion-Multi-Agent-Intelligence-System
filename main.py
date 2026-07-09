import asyncio
from orchestrator.orchestrator import Orchestrator


async def main():
    orch = Orchestrator()
    print("AI Control Tower Ready.")
    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() in ["exit", "quit"]:
            break
        result = await orch.run(query)
        print("\nFinal Answer:\n")
        print(result["final_answer"])
        print("\nKey Points:", result["final_key_points"])
        print("Confidence:", result["confidence"])


if __name__ == "__main__":
    asyncio.run(main())
