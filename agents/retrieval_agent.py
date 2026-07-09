import aiohttp
import asyncio
from agents.base_agent import BaseAgent
from memory.memory_store import MemoryStore
from utils.logger import get_logger
import os

logger = get_logger("retrieval_agent")

MEMORY = MemoryStore()
MEMORY.add("Rephrase.ai is an AI video generation startup based in India.")
MEMORY.add("Sarvam AI focuses on Indian-language LLM development.")
MEMORY.add("Niramai uses AI for thermal-based breast cancer screening.")
MEMORY.add("Observe.ai builds enterprise voice AI technology.")


class RetrievalAgent(BaseAgent):
    name = "RetrievalAgent"
    type = "retrieval"

    def __init__(self):
        super().__init__()

        # Existing APIs
        self.SERPAPI_KEY = os.getenv("SERPAPI_KEY")
        self.GNEWS_KEY = os.getenv("GNEWS_KEY")
        self.SCRAPERAPI_KEY = os.getenv("SCRAPERAPI_KEY")
        self.HF_API_KEY = os.getenv("HF_API_KEY")

        # New free retrieval APIs (ENV variables)
        self.VOYAGE_KEY = os.getenv("VOYAGE_API_KEY")
        self.GROQ_KEY = os.getenv("GROQ_API_KEY")
        self.CF_KEY = os.getenv("CLOUDFLARE_API_KEY")
        self.CF_ACCOUNT = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")

    async def handle(self, task):
        query = task.get("query")

        results = await self.multi_source_search(query)
        ranked = self.rank_results(results)

        return {
            "query": query,
            "results": ranked[:20],
            "sources_used": [r["source"] for r in ranked[:20]],
        }

    # ============================================================
    # PARALLEL MULTI-SOURCE RETRIEVAL
    # ============================================================
    async def multi_source_search(self, query):
        async with aiohttp.ClientSession() as session:
            tasks = [
                # Original APIs
                self.search_serpapi(session, query),
                self.search_gnews(session, query),
                self.search_wikipedia(session, query),
                self.search_arxiv(session, query),
                self.search_github(session, query),
                self.search_stackoverflow(session, query),
                self.scrape_fallback(session, query),
                # New APIs you requested
                self.search_voyage(query),
                self.search_groq(query),
                self.search_cloudflare(query),
                self.search_llamaapi(query),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

        merged = []
        for res in results:
            if isinstance(res, list):
                merged.extend(res)

        return merged

    # ============================================================
    # -------- EXISTING SEARCH PROVIDERS --------
    # ============================================================

    async def search_serpapi(self, session, query):
        url = "https://serpapi.com/search"
        params = {"q": query, "engine": "google", "api_key": self.SERPAPI_KEY}
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            organic = data.get("organic_results", [])
            return [
                {
                    "source": "serpapi",
                    "title": o["title"],
                    "link": o["link"],
                    "snippet": o.get("snippet"),
                }
                for o in organic
            ]

    async def search_gnews(self, session, query):
        url = f"https://gnews.io/api/v4/search?q={query}&token={self.GNEWS_KEY}&lang=en"
        async with session.get(url) as resp:
            data = await resp.json()
            articles = data.get("articles", [])
            return [
                {
                    "source": "gnews",
                    "title": a["title"],
                    "link": a["url"],
                    "snippet": a["description"],
                }
                for a in articles
            ]

    async def search_wikipedia(self, session, query):
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        async with session.get(url) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return [
                {
                    "source": "wikipedia",
                    "title": data.get("title"),
                    "snippet": data.get("extract"),
                    "link": data.get("content_urls", {}).get("desktop", {}).get("page"),
                }
            ]

    async def search_arxiv(self, session, query):
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
        async with session.get(url) as resp:
            text = await resp.text()
            return [{"source": "arxiv", "snippet": text[:500]}]

    async def search_github(self, session, query):
        url = f"https://api.github.com/search/repositories?q={query}"
        async with session.get(url) as resp:
            data = await resp.json()
            return [
                {
                    "source": "github",
                    "title": i["name"],
                    "snippet": i["description"],
                    "link": i["html_url"],
                }
                for i in data.get("items", [])[:5]
            ]

    async def search_stackoverflow(self, session, query):
        url = "https://api.stackexchange.com/2.3/search"
        params = {
            "order": "desc",
            "sort": "relevance",
            "intitle": query,
            "site": "stackoverflow",
        }
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            return [
                {"source": "stackoverflow", "title": i["title"], "link": i["link"]}
                for i in data.get("items", [])[:5]
            ]

    async def scrape_fallback(self, session, query):
        url = f"http://api.scraperapi.com?api_key={self.SCRAPERAPI_KEY}&url=https://www.google.com/search?q={query}"
        async with session.get(url) as resp:
            html = await resp.text()
            return [{"source": "scraperapi", "snippet": html[:400]}]

    # ============================================================
    # -------- NEW SEARCH PROVIDERS YOU REQUESTED --------
    # ============================================================

    async def search_voyage(self, query):
        """Voyage AI – semantic search (free tier)"""
        if not self.VOYAGE_KEY:
            return []
        url = "https://api.voyageai.com/v1/retrieve"
        headers = {"Authorization": f"Bearer {self.VOYAGE_KEY}"}
        payload = {"query": query, "top_k": 5}

        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(url, json=payload, headers=headers) as r:
                    data = await r.json()
                    return [
                        {"source": "voyage", "snippet": d.get("text")}
                        for d in data.get("results", [])
                    ]
        except:
            return []

    async def search_groq(self, query):
        """Groq Cloud — LLaMA 3 & Mixtral completion-style search"""
        if not self.GROQ_KEY:
            return []
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.GROQ_KEY}"}
        payload = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": f"Search summary: {query}"}],
            "max_tokens": 150,
        }

        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(url, headers=headers, json=payload) as r:
                    data = await r.json()
                    text = data["choices"][0]["message"]["content"]
                    return [{"source": "groq", "snippet": text}]
        except:
            return []

    async def search_cloudflare(self, query):
        """Cloudflare Workers AI – free Llama3 inference"""
        if not self.CF_KEY:
            return []

        url = f"https://api.cloudflare.com/client/v4/accounts/{self.CF_ACCOUNT}/ai/run/@cf/meta/llama-3-8b-instruct"
        headers = {"Authorization": f"Bearer {self.CF_KEY}"}

        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(url, headers=headers, json={"input": query}) as r:
                    data = await r.json()
                    out = data.get("result", {}).get("response", "")
                    return [{"source": "cloudflare", "snippet": out}]
        except:
            return []

    async def search_llamaapi(self, query):
        """LlamaAPI – cloud LLaMA inference"""
        if not self.LLAMA_API_KEY:
            return []

        url = "https://api.llama-api.com/prompt"
        headers = {"Authorization": f"Bearer {self.LLAMA_API_KEY}"}
        payload = {"prompt": query, "model": "llama3-8b"}

        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(url, json=payload, headers=headers) as r:
                    data = await r.json()
                    return [{"source": "llamaapi", "snippet": data.get("response", "")}]
        except:
            return []

    # ============================================================
    # RESULT RANKING
    # ============================================================
    def rank_results(self, results):
        def score(item):
            s = 0

            tier1 = {"serpapi", "gnews", "wikipedia"}
            if item["source"] in tier1:
                s += 3

            if item["source"] == "arxiv":
                s += 2

            if item["source"] in {"github", "stackoverflow"}:
                s += 1

            snippet = item.get("snippet") or item.get("text") or ""
            s += len(snippet) / 300

            return s

        return sorted(results, key=score, reverse=True)
