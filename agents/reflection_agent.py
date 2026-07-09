import os
import aiohttp
from typing import Dict, Any
from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger("reflection_agent")


class ReflectionAgent(BaseAgent):
    name = "ReflectionAgent"
    type = "reflection"

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
        supported = task.get("supported_points", []) or []
        unsupported = task.get("unsupported_points", []) or []
        hallucinations = task.get("hallucinations", []) or []
        contradiction_score = task.get("contradiction_score", 0.0)

        if not summary:
            return {
                "final_answer": "",
                "final_key_points": [],
                "hallucinations_removed": hallucinations,
                "unsupported_fixed": unsupported,
                "contradiction_score": contradiction_score,
            }

        refined = await self._generate_refined_output(
            summary,
            supported,
            unsupported,
            hallucinations,
            contradiction_score,
            evidence,
        )
        final_kp = await self._extract_key_points(refined)
        return {
            "final_answer": refined.strip(),
            "final_key_points": final_kp,
            "hallucinations_removed": hallucinations,
            "unsupported_fixed": unsupported,
            "contradiction_score": contradiction_score,
        }

    async def _generate_refined_output(
        self,
        summary,
        supported,
        unsupported,
        hallucinations,
        contradiction_score,
        evidence,
    ):
        evidence_text = "\n\n".join([e.get("text", "") for e in evidence])
        prompt = f"""You are the Reflection Agent. Fix the summary using evidence, remove unsupported claims and hallucinations.

Original Summary:
{summary}

Supported Points:
{supported}

Unsupported Points:
{unsupported}

Hallucinations:
{hallucinations}

Contradiction Score: {contradiction_score}

Evidence:
{evidence_text}

Return a concise, factual final answer grounded only in the evidence. Output only the final answer."""
        return await self._llm_request(prompt, max_tokens=400)

    async def _extract_key_points(self, text):
        prompt = f"Extract 5 concise key points from the answer below. Return a JSON array.\n\n{text}"
        out = await self._llm_request(prompt, max_tokens=150)
        return self._parse_json_like_array(out)

    async def _llm_request(self, prompt, max_tokens=200):
        if self.GROQ_API_KEY:
            try:
                out = await self._groq_request(prompt, max_tokens)
                if out:
                    return out
            except Exception as e:
                logger.error("Groq reflection error: %s", e)
        if self.CF_API_KEY and self.CF_ACCOUNT_ID:
            try:
                out = await self._cloudflare_request(prompt)
                if out:
                    return out
            except Exception as e:
                logger.error("Cloudflare reflection error: %s", e)
        return await self._ollama_request(prompt, max_tokens)

    async def _groq_request(self, prompt, max_tokens):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.GROQ_API_KEY}"}
        payload = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(url, headers=headers, json=payload, timeout=60) as r:
                data = await r.json()
                return (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                )

    async def _cloudflare_request(self, prompt):
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.CF_ACCOUNT_ID}/ai/run/@cf/meta/llama-3-8b-instruct"
        headers = {"Authorization": f"Bearer {self.CF_API_KEY}"}
        async with aiohttp.ClientSession() as s:
            async with s.post(
                url, headers=headers, json={"input": prompt}, timeout=60
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
            async with s.post(self.OLLAMA_URL, json=payload, timeout=60) as r:
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
            except:
                pass
        lines = [l.strip("-• ").strip() for l in t.split("\n") if l.strip()]
        if lines:
            return lines[:5]
        return [p.strip() for p in t.split(",") if p.strip()][:5]
