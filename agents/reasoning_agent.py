import os
import aiohttp
from typing import List, Dict, Any
from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger("reasoning_agent")


class ReasoningAgent(BaseAgent):
    name = "ReasoningAgent"
    type = "reasoning"

    def __init__(self, memory=None, config=None, tools=None):
        super().__init__(memory=memory, config=config, tools=tools)
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.CF_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
        self.CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
        self.SUMMARY_TOKENS = int(os.getenv("SUMMARY_TOKENS", "512"))

    async def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        query = task.get("query", "")
        retrieved = task.get("results", []) or []
        evidence = self._collect_top_snippets(retrieved, limit=12)
        if not evidence:
            return {
                "query": query,
                "summary": "",
                "key_points": [],
                "entities": [],
                "contradictions": [],
                "evidence": [],
                "confidence": 0.0,
            }
        prompt = self._build_synthesis_prompt(query, evidence)
        synthesis = await self._llm_request(prompt, max_tokens=self.SUMMARY_TOKENS)
        key_points = await self._extract_key_points(synthesis)
        entities = await self._extract_entities(evidence)
        contradictions = await self._find_contradictions(evidence)
        confidence = self._estimate_confidence(evidence, synthesis)
        return {
            "query": query,
            "summary": synthesis.strip(),
            "key_points": key_points,
            "entities": entities,
            "contradictions": contradictions,
            "evidence": evidence,
            "confidence": round(confidence, 3),
        }

    def _collect_top_snippets(
        self, results: List[Dict[str, Any]], limit: int = 12
    ) -> List[Dict[str, Any]]:
        snippets = []
        for item in results:
            text = item.get("snippet") or item.get("content") or item.get("title") or ""
            if not text:
                continue
            snippets.append(
                {
                    "source": item.get("source"),
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "text": text,
                }
            )
            if len(snippets) >= limit:
                break
        return snippets

    def _build_synthesis_prompt(
        self, query: str, evidence: List[Dict[str, Any]]
    ) -> str:
        parts = [
            f"User query: {query}",
            "Synthesize a concise evidence-backed analysis with a short summary and 5 key points.",
        ]
        for i, e in enumerate(evidence, 1):
            parts.append(f"Source {i} ({e.get('source')}): {e.get('text')}")
        parts.append(
            "Produce a JSON with fields: summary, key_points (list), important_entities (list). Do not include sources in the summary but ensure claims are grounded."
        )
        return "\n\n".join(parts)

    async def _llm_request(self, prompt: str, max_tokens: int = 512) -> str:
        # Groq
        if self.GROQ_API_KEY:
            try:
                out = await self._groq_request(prompt, max_tokens)
                if out:
                    return out
            except Exception as e:
                logger.error("Groq error: %s", e)
        # Cloudflare
        if self.CF_API_KEY and self.CF_ACCOUNT_ID:
            try:
                out = await self._cloudflare_request(prompt)
                if out:
                    return out
            except Exception as e:
                logger.error("Cloudflare error: %s", e)
        # Ollama fallback
        return await self._ollama_request(prompt, max_tokens)

    async def _groq_request(self, prompt: str, max_tokens: int):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.GROQ_API_KEY}"}
        payload = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(url, headers=headers, json=payload, timeout=60) as r:
                data = await r.json()
                return (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                )

    async def _cloudflare_request(self, prompt: str):
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.CF_ACCOUNT_ID}/ai/run/@cf/meta/llama-3-8b-instruct"
        headers = {"Authorization": f"Bearer {self.CF_API_KEY}"}
        async with aiohttp.ClientSession() as s:
            async with s.post(
                url, headers=headers, json={"input": prompt}, timeout=60
            ) as r:
                data = await r.json()
                return data.get("result", {}).get("response", "")

    async def _ollama_request(self, prompt: str, max_tokens: int = 512):
        payload = {
            "model": self.LLM_MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": False,
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(self.OLLAMA_URL, json=payload, timeout=60) as r:
                data = await r.json()
                return data.get("response", "")

    async def _extract_key_points(self, synthesis: str) -> List[str]:
        prompt = f"From the text below list the 5 most important actionable key points.\n\nText:\n{synthesis}\n\nReturn as a JSON array of short strings."
        out = await self._llm_request(prompt, max_tokens=200)
        return self._parse_json_like_array(out)

    async def _extract_entities(self, evidence: List[Dict[str, Any]]) -> List[str]:
        combined = "\n\n".join([e["text"] for e in evidence])
        prompt = f"Extract named entities (companies, people, products) from the text below. Return a JSON array of unique names.\n\n{combined}"
        out = await self._llm_request(prompt, max_tokens=200)
        return self._parse_json_like_array(out)

    async def _find_contradictions(self, evidence: List[Dict[str, Any]]) -> List[str]:
        combined = "\n\n".join([f"{e['source']}: {e['text']}" for e in evidence])
        prompt = f"Identify any direct contradictions or conflicting claims across these sources, list them as short descriptions in a JSON array.\n\n{combined}"
        out = await self._llm_request(prompt, max_tokens=200)
        return self._parse_json_like_array(out)

    def _estimate_confidence(self, evidence, synthesis):
        coverage = min(1.0, len(evidence) / 6.0)
        length_bonus = min(1.0, len(synthesis.split()) / 200.0)
        unique_sources = len({e.get("source") for e in evidence})
        diversity = min(1.0, unique_sources / 3.0)
        return 0.25 * coverage + 0.45 * length_bonus + 0.30 * diversity

    def _parse_json_like_array(self, text: str):
        t = (text or "").strip()
        if not t:
            return []
        if t.startswith("[") and t.endswith("]"):
            try:
                import json

                return [str(x) for x in json.loads(t)][:10]
            except Exception:
                pass
        lines = [
            l.strip("- ").strip()
            for l in t.replace("\r", "\n").split("\n")
            if l.strip()
        ]
        if len(lines) > 1:
            return lines[:10]
        return [p.strip() for p in t.split(",") if p.strip()][:10]
