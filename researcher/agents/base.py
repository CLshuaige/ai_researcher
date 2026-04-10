from typing import Optional, Dict, Any, List, Callable
from autogen import ConversableAgent

from researcher.prompts.templates import (
    ASKER_SYSTEM_PROMPT,
    TASK_FORMATTER_SYSTEM_PROMPT,
    LITERATURE_MANAGER_SYSTEM_PROMPT,
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
    CODE_DEBUGGER_SYSTEM_PROMPT,
    ANALYST_SYSTEM_PROMPT,
    PAPER_WRITER_SYSTEM_PROMPT,
    OUTLINER_SYSTEM_PROMPT,
    SECTION_WRITER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
)


from researcher.agents.context_manager import AgentContextManager
from researcher.skill.utils import enable_skill_prompt
from researcher.skill.tools import load_skill, execute_skill_script, load_reference


class BaseAgent:
    """Base class for all research agents"""

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt

    def create_agent(
        self,
        llm_config: Dict[str, Any],
        enable_context_compression: Optional[bool] = False,
        enable_skill: Optional[bool] = False,
        functions: Optional[List[Callable]] = None,
    ) -> ConversableAgent:
        """Create a ConversableAgent with automatic context compression.

        Automatically loads and applies context compression settings from the global YAML configuration.
        Context compression is enabled by default unless explicitly disabled.

        Args:
            llm_config: LLM configuration dictionary.
            enable_context_compression: Whether to enable automatic context compression.
                If None, uses setting from global YAML config (default: True).
                Set to False to disable context compression for this agent.

        Returns:
            ConversableAgent with context compression capabilities applied.
        """
        if functions is None:
            functions = []
        if enable_skill:
            functions.extend([load_skill, load_reference, execute_skill_script])
            skill_prompt = enable_skill_prompt()
            system_prompt = self.system_prompt + "\n\n" + skill_prompt
        else:
            system_prompt = self.system_prompt
            
        agent = ConversableAgent(
            name=self.name,
            system_message=system_prompt,
            llm_config=llm_config,
            functions=functions,
        )

        global_config = None
        try:
            from researcher.utils import load_global_config
            global_config = load_global_config()
        except Exception:
            pass

        should_enable_compression = enable_context_compression

        if global_config and should_enable_compression is not False:
            context_config = global_config.get("researcher", {}).get("context_management", {})

            # Use global setting if not explicitly overridden
            if should_enable_compression is None:
                should_enable_compression = context_config.get("enable_compression", False)

            # Apply compression if enabled
            if should_enable_compression:
                AgentContextManager.create_from_config(context_config, agent, llm_config)

                # Apply message history limiting if configured
                message_history_config = context_config.get("message_history", {})
                if message_history_config.get("enable_history_limiting", True):
                    AgentContextManager.apply_message_history_limiting(agent, message_history_config)

        return agent


# Task Parsing Module
class AskerAgent(BaseAgent):
    def __init__(self, name: str = "Asker"):
        super().__init__(name, ASKER_SYSTEM_PROMPT)


class TaskFormatterAgent(BaseAgent):
    def __init__(self, name: str = "TaskFormatter"):
        super().__init__(name, TASK_FORMATTER_SYSTEM_PROMPT)


# Literature Review Module
class LiteratureManagerAgent(BaseAgent):
    def __init__(self, name: str = "LiteratureManager"):
        super().__init__(name, LITERATURE_MANAGER_SYSTEM_PROMPT)

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


class CodeDebuggerAgent(BaseAgent):
    def __init__(self, name: str = "CodeDebugger"):
        super().__init__(name, CODE_DEBUGGER_SYSTEM_PROMPT)


class AnalystAgent(BaseAgent):
    def __init__(self, name: str = "Analyst"):
        super().__init__(name, ANALYST_SYSTEM_PROMPT)


# Report Generation Module
class PaperWriterAgent(BaseAgent):
    def __init__(self, name: str = "PaperWriter"):
        super().__init__(name, PAPER_WRITER_SYSTEM_PROMPT)

class OutlinerAgent(BaseAgent):
    def __init__(self, name: str = "Outliner"):
        super().__init__(name, OUTLINER_SYSTEM_PROMPT)

class SectionWriterAgent(BaseAgent):
    def __init__(self, name: str = "SectionWriter"):
        super().__init__(name, SECTION_WRITER_SYSTEM_PROMPT)


# Review Module
class ReviewerAgent(BaseAgent):
    def __init__(self, name: str = "Reviewer"):
        super().__init__(name, REVIEWER_SYSTEM_PROMPT)
