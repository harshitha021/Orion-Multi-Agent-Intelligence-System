import uuid
from agents.retrieval_agent import RetrievalAgent
from agents.reasoning_agent import ReasoningAgent
from agents.audit_agent import AuditAgent
from agents.reflection_agent import ReflectionAgent
from agents.router_agent import RouterAgent
from utils.logger import get_logger

logger = get_logger("orchestrator")


class Orchestrator:
    def __init__(self):
        self.retrieval = RetrievalAgent()
        self.reasoning = ReasoningAgent()
        self.audit = AuditAgent()
        self.reflection = ReflectionAgent()
        self.router = RouterAgent()

    async def run(self, query: str):
        task_id = str(uuid.uuid4())
        route = await self.router.handle({"id": task_id, "query": query})
        route_type = route.get("route", "retrieval_required")
        base = {
            "query": query,
            "retrieval": None,
            "reasoning": None,
            "audit": None,
            "final_answer": "",
            "final_key_points": [],
            "confidence": 0.0,
        }
        if route_type == "greeting":
            base.update(
                {
                    "final_answer": "Hello! How can I assist you today?",
                    "confidence": 1.0,
                }
            )
            return base
        if route_type == "direct_answer":
            # For short queries, attempt reasoning-only run
            reasoning = await self.reasoning.handle(
                {"id": task_id, "query": query, "results": []}
            )
            base["reasoning"] = reasoning
            base["final_answer"] = reasoning.get("summary", "")
            base["final_key_points"] = reasoning.get("key_points", [])
            base["confidence"] = reasoning.get("confidence", 0.0)
            return base
        # Full pipeline
        retrieved = await self.retrieval.handle({"id": task_id, "query": query})
        base["retrieval"] = retrieved
        results = retrieved.get("results", []) if isinstance(retrieved, dict) else []
        reasoning = await self.reasoning.handle(
            {"id": task_id, "query": query, "results": results}
        )
        base["reasoning"] = reasoning
        if not reasoning.get("summary"):
            base["final_answer"] = "Sorry, I couldn't generate a reliable answer."
            base["confidence"] = 0.0
            return base
        audit = await self.audit.handle(
            {
                "id": task_id,
                "summary": reasoning.get("summary"),
                "key_points": reasoning.get("key_points"),
                "entities": reasoning.get("entities"),
                "contradictions": reasoning.get("contradictions"),
                "evidence": reasoning.get("evidence"),
            }
        )
        base["audit"] = audit
        reflection = await self.reflection.handle(
            {
                "id": task_id,
                "summary": reasoning.get("summary"),
                "key_points": reasoning.get("key_points"),
                "evidence": reasoning.get("evidence"),
                "supported_points": audit.get("supported_points"),
                "unsupported_points": audit.get("unsupported_points"),
                "hallucinations": audit.get("hallucinations"),
                "contradiction_score": audit.get("contradiction_score"),
            }
        )
        base["final_answer"] = reflection.get("final_answer") or reasoning.get(
            "summary"
        )
        base["final_key_points"] = reflection.get("final_key_points") or reasoning.get(
            "key_points"
        )
        base["confidence"] = reasoning.get("confidence", 0.0)
        return base
