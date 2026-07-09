import os
import aiohttp
from typing import Dict, Any
from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger("router_agent")


class RouterAgent(BaseAgent):
    name = "RouterAgent"
    type = "router"

    def __init__(self, memory=None, config=None, tools=None):
        super().__init__(memory=memory, config=config, tools=tools)
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.CF_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
        self.CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
        self.greetings = {"hi", "hello", "hey", "yo", "hii", "hiii", "sup"}

    async def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        query = (task.get("query") or "").strip().lower()
        if not query:
            return {"route": "invalid"}
        if query in self.greetings:
            return {"route": "greeting"}
        rb = self._rule_based_route(query)
        if rb:
            return rb
        category = await self._llm_classify(query)
        return {"route": category}

    def _rule_based_route(self, q: str):
        if any(w in q for w in ["define", "meaning", "what is"]):
            return {"route": "reasoning_only"}
        if any(w in q for w in ["python", "code", "fix"]):
            return {"route": "coding"}
        if any(w in q for w in ["math", "calculate", "solve"]):
            return {"route": "math"}
        if len(q.split()) <= 2:
            return {"route": "direct_answer"}
        return None

    async def _llm_classify(self, query: str):
        prompt = f"Classify the query into one category: greeting, smalltalk, coding, math, factual_query, reasoning_required, retrieval_required. Return only the category.\n\nQuery: {query}"
        out = (await self._llm_request(prompt)).strip().lower()
        valid = {
            "greeting",
            "smalltalk",
            "coding",
            "math",
            "factual_query",
            "reasoning_required",
            "retrieval_required",
        }
        return out if out in valid else "retrieval_required"

    async def _llm_request(self, prompt: str):
        # try Groq
        if self.GROQ_API_KEY:
            try:
                out = await self._groq_request(prompt)
                if out:
                    return out
            except Exception as e:
                logger.error("Groq classify error: %s", e)
        # try Cloudflare
        if self.CF_API_KEY and self.CF_ACCOUNT_ID:
            try:
                out = await self._cloudflare_request(prompt)
                if out:
                    return out
            except Exception as e:
                logger.error("Cloudflare classify error: %s", e)
        # fallback Ollama
        try:
            return await self._ollama_request(prompt)
        except Exception as e:
            logger.error("Ollama classify error: %s", e)
            return "retrieval_required"

    async def _groq_request(self, prompt: str):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.GROQ_API_KEY}"}
        payload = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(url, headers=headers, json=payload, timeout=30) as r:
                data = await r.json()
                return (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                )

    async def _cloudflare_request(self, prompt: str):
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.CF_ACCOUNT_ID}/ai/run/@cf/meta/llama-3-8b-instruct"
        headers = {"Authorization": f"Bearer {self.CF_API_KEY}"}
        async with aiohttp.ClientSession() as s:
            async with s.post(
                url, headers=headers, json={"input": prompt}, timeout=30
            ) as r:
                data = await r.json()
                return data.get("result", {}).get("response", "")

    async def _ollama_request(self, prompt: str):
        payload = {"model": self.LLM_MODEL, "prompt": prompt, "stream": False}
        async with aiohttp.ClientSession() as s:
            async with s.post(self.OLLAMA_URL, json=payload, timeout=30) as r:
                data = await r.json()
                return data.get("response", "")
