from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from autogen import AssistantAgent
from typing import Dict, Any

from researcher.prompts.templates import (
    ASKER_SYSTEM_PROMPT,
    TASK_FORMATTER_SYSTEM_PROMPT,
    LITERATURE_SEARCHER_SYSTEM_PROMPT,
    LITERATURE_SUMMARIZER_SYSTEM_PROMPT,
    IDEA_PROPOSER_SYSTEM_PROMPT,
    IDEA_CRITIC_SYSTEM_PROMPT,
    IDEA_FORMATTER_SYSTEM_PROMPT,
    METHOD_PLANNER_SYSTEM_PROMPT,
    METHOD_CRITIC_SYSTEM_PROMPT,
    METHOD_FORMATTER_SYSTEM_PROMPT,
    RA_SYSTEM_PROMPT,
    ENGINEER_SYSTEM_PROMPT,
    ANALYST_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
)


class BaseAgent:
    """Base class for all research agents"""

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt

    def create_assistant(self, llm_config: Dict[str, Any]) -> AssistantAgent:
        return AssistantAgent(
            name=self.name,
            system_message=self.system_prompt,
            llm_config=llm_config,
            human_input_mode="NEVER"
        )


# Task Parsing Module
class AskerAgent(BaseAgent):
    def __init__(self, name: str = "Asker"):
        super().__init__(name, ASKER_SYSTEM_PROMPT)


class TaskFormatterAgent(BaseAgent):
    def __init__(self, name: str = "TaskFormatter"):
        super().__init__(name, TASK_FORMATTER_SYSTEM_PROMPT)


# Literature Review Module
class LiteratureSearcherAgent(BaseAgent):
    def __init__(self, name: str = "LiteratureSearcher"):
        super().__init__(name, LITERATURE_SEARCHER_SYSTEM_PROMPT)


class LiteratureSummarizerAgent(BaseAgent):
    def __init__(self, name: str = "LiteratureSummarizer"):
        super().__init__(name, LITERATURE_SUMMARIZER_SYSTEM_PROMPT)


# Hypothesis Construction Module
class IdeaProposerAgent(BaseAgent):
    def __init__(self, name: str = "IdeaProposer"):
        super().__init__(name, IDEA_PROPOSER_SYSTEM_PROMPT)


class IdeaCriticAgent(BaseAgent):
    def __init__(self, name: str = "IdeaCritic"):
        super().__init__(name, IDEA_CRITIC_SYSTEM_PROMPT)


class IdeaFormatterAgent(BaseAgent):
    def __init__(self, name: str = "IdeaFormatter"):
        super().__init__(name, IDEA_FORMATTER_SYSTEM_PROMPT)


# Method Design Module
class MethodPlannerAgent(BaseAgent):
    def __init__(self, name: str = "MethodPlanner"):
        super().__init__(name, METHOD_PLANNER_SYSTEM_PROMPT)


class MethodCriticAgent(BaseAgent):
    def __init__(self, name: str = "MethodCritic"):
        super().__init__(name, METHOD_CRITIC_SYSTEM_PROMPT)


class MethodFormatterAgent(BaseAgent):
    def __init__(self, name: str = "MethodFormatter"):
        super().__init__(name, METHOD_FORMATTER_SYSTEM_PROMPT)


# Experiment Execution Module
class RAAgent(BaseAgent):
    def __init__(self, name: str = "RA"):
        super().__init__(name, RA_SYSTEM_PROMPT)


class EngineerAgent(BaseAgent):
    def __init__(self, name: str = "Engineer"):
        super().__init__(name, ENGINEER_SYSTEM_PROMPT)


class AnalystAgent(BaseAgent):
    def __init__(self, name: str = "Analyst"):
        super().__init__(name, ANALYST_SYSTEM_PROMPT)


# Report Generation Module
class WriterAgent(BaseAgent):
    def __init__(self, name: str = "Writer"):
        super().__init__(name, WRITER_SYSTEM_PROMPT)


# Review Module
class ReviewerAgent(BaseAgent):
    def __init__(self, name: str = "Reviewer"):
        super().__init__(name, REVIEWER_SYSTEM_PROMPT)
