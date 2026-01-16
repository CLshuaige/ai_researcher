from typing import Dict, Any
from pathlib import Path

from autogen import GroupChat, GroupChatManager, UserProxyAgent

from researcher.state import ResearchState
from researcher.schemas import ResearchIdea, IdeaCandidate
from researcher.agents import IdeaProposerAgent, IdeaCriticAgent, IdeaFormatterAgent
from researcher.config import DEBATE_MAX_ROUNDS
from researcher.utils import (
    save_markdown,
    log_stage,
    get_artifact_path,
    load_artifact_from_file,
    get_llm_config,
    parse_json_from_response,
)
from researcher.prompts.templates import IDEA_PROPOSAL_PROMPT, IDEA_FORMATTER_PROMPT
from researcher.exceptions import WorkflowError


def hypothesis_construction_node(state: ResearchState) -> Dict[str, Any]:
    """Construct research hypothesis through multi-agent debate"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "hypothesis_construction", "Starting hypothesis construction")

    try:
        task = load_artifact_from_file(workspace_dir, "task")
        literature_text = load_artifact_from_file(workspace_dir, "literature") or "No literature review available"

        if not task:
            raise WorkflowError("Task file not found")

        llm_config = get_llm_config()

        proposer = IdeaProposerAgent().create_assistant(llm_config)
        critic = IdeaCriticAgent().create_assistant(llm_config)

        initial_message = IDEA_PROPOSAL_PROMPT.format(task=task, literature=literature_text)

        log_stage(workspace_dir, "hypothesis_construction", f"Running debate (max {DEBATE_MAX_ROUNDS} rounds)")

        groupchat = GroupChat(
            agents=[proposer, critic],
            messages=[],
            max_round=DEBATE_MAX_ROUNDS,
            speaker_selection_method="round_robin"
        )

        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
        proposer.initiate_chat(manager, message=initial_message)

        debate_history = _extract_history(groupchat.messages)
        _save_log(workspace_dir, debate_history, "idea_debate")

        log_stage(workspace_dir, "hypothesis_construction", "Formatting and evaluating ideas")
        formatted_output = _format_output(debate_history, llm_config)

        idea = _parse_idea(formatted_output, len(debate_history) // 2)

        idea_path = get_artifact_path(workspace_dir, "idea")
        save_markdown(idea.to_markdown(), idea_path)

        log_stage(workspace_dir, "hypothesis_construction",
                 f"Completed. Generated {len(idea.candidates)} ideas")

        return {"task": task, "idea": idea, "stage": "hypothesis_construction"}

    except Exception as e:
        log_stage(workspace_dir, "hypothesis_construction", f"Error: {str(e)}")
        raise WorkflowError(f"Hypothesis construction failed: {str(e)}")


def _extract_history(messages: list) -> list[Dict[str, str]]:
    history = []
    for msg in messages:
        if isinstance(msg, dict) and "content" in msg and "name" in msg:
            history.append({"role": msg["name"], "content": msg["content"]})
    return history


def _save_log(workspace_dir: Path, history: list[Dict[str, str]], log_name: str):
    log_dir = workspace_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{log_name}.log", 'w', encoding='utf-8') as f:
        for entry in history:
            f.write(f"[{entry['role']}]\n{entry['content']}\n\n")


def _format_output(debate_history: list[Dict[str, str]], llm_config: Dict[str, Any]) -> Dict[str, Any]:
    history_text = "\n\n".join([f"**{e['role']}**: {e['content']}" for e in debate_history])
    prompt = IDEA_FORMATTER_PROMPT.format(debate_history=history_text)

    formatter = IdeaFormatterAgent().create_assistant(llm_config)
    user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=0)
    user_proxy.initiate_chat(formatter, message=prompt)

    response = user_proxy.last_message()["content"]

    return parse_json_from_response(response)


def _parse_idea(formatted_output: Dict[str, Any], debate_rounds: int) -> ResearchIdea:
    candidates = []
    for idea_data in formatted_output.get("ideas", []):
        candidate = IdeaCandidate(
            content=idea_data.get("content", ""),
            score=idea_data.get("score", 0.0),
            round=idea_data.get("round", 0),
            criticisms=idea_data.get("weaknesses", [])
        )
        candidates.append(candidate)

    return ResearchIdea(
        candidates=candidates,
        selected_index=0,  # First one is best (already ranked by FormatterAgent)
        debate_rounds=debate_rounds
    )
