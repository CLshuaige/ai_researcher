from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json

from autogen import AssistantAgent, GroupChat, GroupChatManager

from researcher.config import config
from researcher.agents import ProposerAgent, CriticAgent, FormatterAgent
from researcher.llm import get_llm_client
from researcher.exceptions import DebateError


@dataclass
class DebateResult:
    """Result of a debate session"""
    final_output: Dict[str, Any]
    debate_history: List[Dict[str, str]]
    rounds: int
    success: bool


class DebateTeam:
    """Manages multi-agent debate between proposer and critic agents"""

    def __init__(
        self,
        proposer: ProposerAgent,
        critic: CriticAgent,
        max_rounds: int,
        workspace_dir: Optional[Path] = None
    ):
        self.proposer = proposer
        self.critic = critic
        self.max_rounds = max_rounds
        self.workspace_dir = workspace_dir
        self._init_assistant_agents()

    def _init_assistant_agents(self):
        """Initialize proposer and critic AssistantAgent instances"""
        llm_config = {
            "model": config.model.model_name,
            "api_key": config.model.api_key,
            "temperature": config.model.temperature,
            "max_tokens": config.model.max_tokens
        }

        if config.model.base_url:
            llm_config["base_url"] = config.model.base_url

        self.proposer_agent = self.proposer.create_assistant_agent(llm_config)
        self.critic_agent = self.critic.create_assistant_agent(llm_config)

    def run(self, initial_message: str) -> DebateResult:
        """
        Run debate between proposer and critic

        Args:
            initial_message: Initial task/context for the debate

        Returns:
            DebateResult with formatted output and history
        """
        try:
            groupchat = GroupChat(
                agents=[self.proposer_agent, self.critic_agent],
                messages=[],
                max_round=self.max_rounds,
                speaker_selection_method="round_robin"
            )

            manager = GroupChatManager(
                groupchat=groupchat,
                llm_config={
                    "model": config.model.model_name,
                    "api_key": config.model.api_key,
                    "temperature": config.model.temperature
                }
            )

            self.proposer_agent.initiate_chat(
                manager,
                message=initial_message
            )

            debate_history = self._extract_history(groupchat.messages)

            if self.workspace_dir:
                self._save_debate_history(debate_history)

            formatted_output = self._format_output(debate_history)

            return DebateResult(
                final_output=formatted_output,
                debate_history=debate_history,
                rounds=len(debate_history) // 2,  # Approximate rounds
                success=True
            )

        except Exception as e:
            raise DebateError(f"Debate failed: {str(e)}")

    def _extract_history(self, messages: List[Dict]) -> List[Dict[str, str]]:
        """Extract clean debate history from group chat messages"""
        history = []
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg and "name" in msg:
                history.append({
                    "role": msg["name"],
                    "content": msg["content"]
                })
        return history

    def _save_debate_history(self, history: List[Dict[str, str]]):
        """Save debate history to log file"""
        log_dir = self.workspace_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{self.proposer.role}_debate.log"

        with open(log_file, 'w', encoding='utf-8') as f:
            for entry in history:
                f.write(f"[{entry['role']}]\n{entry['content']}\n\n")

    def _format_output(self, debate_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Format debate output using FormatterAgent with LLM"""
        from researcher.prompts.templates import IDEA_FORMATTER_PROMPT, METHOD_FORMATTER_PROMPT
        from researcher.llm import get_llm_client
        import json

        # Determine which formatter prompt to use based on proposer role
        if "idea" in self.proposer.role.lower() or "hypothesis" in self.proposer.role.lower():
            formatter_prompt = IDEA_FORMATTER_PROMPT
        else:
            formatter_prompt = METHOD_FORMATTER_PROMPT

        history_text = "\n\n".join([
            f"**{entry['role']}**: {entry['content']}"
            for entry in debate_history
        ])

        prompt = formatter_prompt.format(debate_history=history_text)

        llm_client = get_llm_client(config.model)

        messages = [
            {"role": "system", "content": FormatterAgent().system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = llm_client.generate(messages)

        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            formatted_output = json.loads(json_str)
            return formatted_output
        except json.JSONDecodeError as e:
            raise DebateError(f"Failed to parse formatter output: {str(e)}\nResponse: {response}")
