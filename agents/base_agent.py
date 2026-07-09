import time
from typing import Dict, Any

from utils.logger import get_logger

logger = get_logger("base_agent")


class BaseAgent:
    name = "BaseAgent"
    type = "base"

    def __init__(self, memory=None, config=None, tools=None):
        self.memory = memory
        self.config = config
        self.tools = tools

    def handle(self, task):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def run_and_push(self, task):
        task_id = task.get("id", "default_id")
        logger.info(f"Agent {self.name} started handling task {task_id}")
        start_time = time.time()
        try:
            payload = self.handle(task)
            result = {
                "agent": self.name,
                "agent_type": self.type,
                "result": payload,
                "task_id": task_id,
                "duration": time.time() - start_time,
            }
            logger.info(
                f"Agent {self.name} finished handling task {task_id} in {result['duration']:.2f} seconds"
            )
        except Exception as e:
            logger.error(
                f"Agent {self.name} encountered an error while handling task {task_id}: {e}"
            )
            result = {
                "error": str(e),
                "task_id": task_id,
                "agent": self.name,
                "agent_type": self.type,
                "duration": time.time() - start_time,
            }

        return result
