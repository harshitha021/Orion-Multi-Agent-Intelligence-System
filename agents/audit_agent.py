import os
import aiohttp
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger("audit_agent")


class AuditAgent(BaseAgent):
    name = "AuditAgent"
    type = "audit"

    def __init__(self, memory=None, config=None, tools=None):
        super().__init__(memory=memory, config=config, tools=tools)
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.CF_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
        self.CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

    async def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        summary = task.get("summary", "")
        key_points = task.get("key_points", []) or []
        evidence = task.get("evidence", []) or []
        if not summary or not evidence:
            return {
                "supported_points": [],
                "unsupported_points": [],
                "hallucinations": [],
                "contradiction_score": 0.0,
                "factual_accuracy": 0.0,
            }
        supported, unsupported = await self._check_support_for_points(
            key_points, evidence
        )
        hallucinations = await self._detect_hallucinations(summary, evidence)
        contradiction_score = self._score_contradictions(
            await self._find_contradictions(evidence)
        )
        factual_accuracy = self._compute_factual_accuracy(
            supported, unsupported, contradiction_score
        )
        return {
            "supported_points": supported,
            "unsupported_points": unsupported,
            "hallucinations": hallucinations,
            "contradiction_score": contradiction_score,
            "factual_accuracy": factual_accuracy,
        }

    async def _check_support_for_points(
        self, key_points: List[str], evidence: List[Dict[str, Any]]
    ):
        supported, unsupported = [], []
        combined = "\n\n".join([e["text"] for e in evidence])
        for point in key_points:
            prompt = f"Does the evidence support this claim? Answer YES or NO only.\n\nClaim: {point}\n\nEvidence:\n{combined}"
            out = (await self._llm_request(prompt)).strip().lower()
            if "yes" in out:
                supported.append(point)
            else:
                unsupported.append(point)
        return supported, unsupported

    async def _detect_hallucinations(
        self, summary: str, evidence: List[Dict[str, Any]]
    ):
        combined = "\n\n".join([e["text"] for e in evidence])
        prompt = f"List statements in the summary that are NOT supported by the evidence. Return a JSON array.\n\nSummary:\n{summary}\n\nEvidence:\n{combined}"
        out = await self._llm_request(prompt)
        return self._parse_json_like_array(out)

    async def _find_contradictions(self, evidence):
        combined = "\n\n".join([f"{e['source']}: {e['text']}" for e in evidence])
        prompt = f"Identify contradictions across these sources. Return JSON array of short descriptions.\n\n{combined}"
        out = await self._llm_request(prompt)
        return self._parse_json_like_array(out)

    def _score_contradictions(self, contradictions: List[str]) -> float:
        if not contradictions:
            return 0.0
        return min(1.0, len(contradictions) / 5.0)

    def _compute_factual_accuracy(self, supported, unsupported, contradiction_score):
        total = len(supported) + len(unsupported)
        if total == 0:
            return 0.0
        base = len(supported) / total
        penalty = contradiction_score * 0.4
        score = base - penalty
        return round(max(0.0, min(1.0, score)), 3)

    async def _llm_request(self, prompt: str):
        if self.GROQ_API_KEY:
            try:
                out = await self._groq_request(prompt)
                if out:
                    return out
            except Exception as e:
                logger.error("Groq audit error: %s", e)
        if self.CF_API_KEY and self.CF_ACCOUNT_ID:
            try:
                out = await self._cloudflare_request(prompt)
                if out:
                    return out
            except Exception as e:
                logger.error("Cloudflare audit error: %s", e)
        return await self._ollama_request(prompt)

    async def _groq_request(self, prompt, max_tokens=200):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.GROQ_API_KEY}"}
        payload = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(url, headers=headers, json=payload, timeout=30) as r:
                data = await r.json()
                return (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                )

    async def _cloudflare_request(self, prompt):
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.CF_ACCOUNT_ID}/ai/run/@cf/meta/llama-3-8b-instruct"
        headers = {"Authorization": f"Bearer {self.CF_API_KEY}"}
        async with aiohttp.ClientSession() as s:
            async with s.post(
                url, headers=headers, json={"input": prompt}, timeout=30
            ) as r:
                data = await r.json()
                return data.get("result", {}).get("response", "")

    async def _ollama_request(self, prompt, max_tokens=200):
        payload = {
            "model": self.LLM_MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": False,
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(self.OLLAMA_URL, json=payload, timeout=30) as r:
                data = await r.json()
                return data.get("response", "")

    def _parse_json_like_array(self, text: str):
        t = (text or "").strip()
        if not t:
            return []
        if t.startswith("[") and t.endswith("]"):
            try:
                import json

                return json.loads(t)
            except Exception:
                pass
        lines = [l.strip("-• ").strip() for l in t.split("\n") if l.strip()]
        if len(lines) > 0:
            return lines[:10]
        return [p.strip() for p in t.split(",") if p.strip()][:10]
