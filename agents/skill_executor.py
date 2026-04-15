"""
SkillExecutor — Iterates over planned tasks and executes them dynamically.
"""

from typing import Any, Dict, List

from core.models import SkillInput
from utils.logger import get_logger

logger = get_logger(__name__)


class SkillExecutor:
    """
    Dynamically executes requested tasks through a dictionary of initialized skills.
    """

    def __init__(self, skills: Dict[str, Any]):
        self.skills = skills
        self.logger = logger

    def execute(self, tasks: List[str], context: dict) -> Dict[str, Any]:
        """
        Executes the planned tasks and manages dynamic state pass-through.
        Returns a dictionary mapping the task name to its output.
        """
        results = {}
        
        # dynamic_state is primed with the incoming context
        dynamic_state = dict(context)

        for task in tasks:
            if task in self.skills:
                self.logger.info(f"[Executor] Running: {task}")
                skill = self.skills[task]
                
                # Execute the skill with the current state payload
                out = skill.safe_execute(SkillInput(data=dynamic_state))
                
                # Store total timing and results
                results[f"{task}_ms"] = out.duration_ms
                results[f"{task}_success"] = out.success
                results[f"{task}_error"] = out.error
                results[f"{task}_data"] = out.data if out.success else None

                # Dynamically update state for downstream skills
                if out.success and out.data:
                    # Generic ParsedDocument propagation for all readers and cleaners
                    if hasattr(out.data, "full_text"):
                        dynamic_state["parsed_document"] = out.data
                        dynamic_state["full_text"] = out.data.full_text

                    # Classification data propagation
                    if task == "document_classifier":
                        dynamic_state["doc_type"] = out.data.doc_type
                        dynamic_state["domain"] = out.data.domain
            else:
                self.logger.warning(f"[Executor] Task requested but not initialized: {task}")

        return results
