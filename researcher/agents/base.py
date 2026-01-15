from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from autogen import AssistantAgent

from researcher.prompts.templates import (
    PROPOSER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    SEARCHER_SYSTEM_PROMPT,
    SUMMARIZER_SYSTEM_PROMPT,
    ENGINEER_SYSTEM_PROMPT,
    RESEARCH_ASSISTANT_SYSTEM_PROMPT,
    ANALYST_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    FORMATTER_SYSTEM_PROMPT
)


class BaseAgent(ABC):
    """Base class for all research agents"""

    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt

    @abstractmethod
    def generate_response(self, context: Dict[str, Any]) -> str:
        """Generate response based on context"""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration for AG2"""
        return {
            "name": self.name,
            "system_message": self.system_prompt,
            "human_input_mode": "NEVER"
        }

    def create_assistant_agent(self, llm_config: Dict[str, Any]) -> AssistantAgent:
        """Instantiate an AssistantAgent for this role"""
        return AssistantAgent(
            name=self.name,
            system_message=self.system_prompt,
            llm_config=llm_config,
            human_input_mode="NEVER"
        )


class ProposerAgent(BaseAgent):
    """Agent for proposing research ideas"""

    def __init__(self, name: str = "Proposer"):
        super().__init__(name, "proposer", PROPOSER_SYSTEM_PROMPT)

    def generate_response(self, context: Dict[str, Any]) -> str:
        # TODO: Implement with LLM client
        pass


class PlannerAgent(BaseAgent):
    """Agent for planning experimental methods"""

    def __init__(self, name: str = "Planner"):
        super().__init__(name, "planner", PLANNER_SYSTEM_PROMPT)

    def generate_response(self, context: Dict[str, Any]) -> str:
        # TODO: Implement with LLM client
        pass


class CriticAgent(BaseAgent):
    """Agent for critiquing proposals"""

    def __init__(self, name: str = "Critic"):
        super().__init__(name, "critic", CRITIC_SYSTEM_PROMPT)

    def generate_response(self, context: Dict[str, Any]) -> str:
        # TODO: Implement with LLM client
        pass


class SearcherAgent(BaseAgent):
    """Agent for literature search"""

    def __init__(self, name: str = "Searcher"):
        super().__init__(name, "searcher", SEARCHER_SYSTEM_PROMPT)

    def generate_response(self, context: Dict[str, Any]) -> str:
        # TODO: Implement with LLM client
        pass


class SummarizerAgent(BaseAgent):
    """Agent for summarizing literature"""

    def __init__(self, name: str = "Summarizer"):
        super().__init__(name, "summarizer", SUMMARIZER_SYSTEM_PROMPT)

    def generate_response(self, context: Dict[str, Any]) -> str:
        # TODO: Implement with LLM client
        pass


class EngineerAgent(BaseAgent):
    """Agent for code implementation"""

    def __init__(self, name: str = "Engineer"):
        super().__init__(name, "engineer", ENGINEER_SYSTEM_PROMPT)

    def generate_response(self, context: Dict[str, Any]) -> str:
        # TODO: Implement with LLM client
        pass


class ResearchAssistantAgent(BaseAgent):
    """Agent for non-code research tasks"""

    def __init__(self, name: str = "ResearchAssistant"):
        super().__init__(name, "research_assistant", RESEARCH_ASSISTANT_SYSTEM_PROMPT)

    def generate_response(self, context: Dict[str, Any]) -> str:
        # TODO: Implement with LLM client
        pass


class AnalystAgent(BaseAgent):
    """Agent for result analysis"""

    def __init__(self, name: str = "Analyst"):
        super().__init__(name, "analyst", ANALYST_SYSTEM_PROMPT)

    def generate_response(self, context: Dict[str, Any]) -> str:
        # TODO: Implement with LLM client
        pass


class WriterAgent(BaseAgent):
    """Agent for paper writing"""

    def __init__(self, name: str = "Writer"):
        super().__init__(name, "writer", WRITER_SYSTEM_PROMPT)

    def generate_response(self, context: Dict[str, Any]) -> str:
        # TODO: Implement with LLM client
        pass


class ReviewerAgent(BaseAgent):
    """Agent for paper review"""

    def __init__(self, name: str = "Reviewer"):
        super().__init__(name, "reviewer", REVIEWER_SYSTEM_PROMPT)

    def generate_response(self, context: Dict[str, Any]) -> str:
        # TODO: Implement with LLM client
        pass


class FormatterAgent(BaseAgent):
    """Agent for formatting and evaluating debate output"""

    def __init__(self, name: str = "Formatter"):
        super().__init__(name, "formatter", FORMATTER_SYSTEM_PROMPT)

    def generate_response(self, context: Dict[str, Any]) -> str:
        # TODO: Implement with LLM client
        pass
