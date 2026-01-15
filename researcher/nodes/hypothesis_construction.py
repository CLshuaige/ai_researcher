from typing import Dict, Any
from pathlib import Path

from researcher.state import ResearchState
from researcher.schemas import ResearchIdea, IdeaCandidate
from researcher.agents import ProposerAgent, CriticAgent
from researcher.debate import DebateTeam
from researcher.config import DEBATE_MAX_ROUNDS
from researcher.utils import save_markdown, log_stage, get_artifact_path, load_artifact_from_file
from researcher.prompts.templates import IDEA_PROPOSAL_PROMPT
from researcher.exceptions import WorkflowError


def hypothesis_construction_node(state: ResearchState) -> Dict[str, Any]:
    """
    Construct research hypothesis through multi-agent debate

    Process:
    1. Initialize Proposer and Critic agents
    2. Run debate for max_rounds
    3. FormatterAgent evaluates and ranks all proposed ideas
    4. Create ResearchIdea object with ranked candidates
    5. Save to workspace
    """
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "hypothesis_construction", "Starting hypothesis construction")

    try:
        task = load_artifact_from_file(workspace_dir, "task")
        literature_text = load_artifact_from_file(workspace_dir, "literature") or "No literature review available"

        if not task:
            raise WorkflowError("Task file not found")

        initial_message = IDEA_PROPOSAL_PROMPT.format(
            task=task,
            literature=literature_text
        )

        proposer = ProposerAgent(name="IdeaProposer")
        critic = CriticAgent(name="IdeaCritic")

        debate_team = DebateTeam(
            proposer=proposer,
            critic=critic,
            max_rounds=DEBATE_MAX_ROUNDS,
            workspace_dir=workspace_dir
        )

        # Run debate
        log_stage(workspace_dir, "hypothesis_construction", f"Running debate (max {DEBATE_MAX_ROUNDS} rounds)")
        debate_result = debate_team.run(initial_message)

        idea = _parse_idea_from_debate(debate_result.final_output, debate_result.rounds)

        idea_path = get_artifact_path(workspace_dir, "idea")
        save_markdown(idea.to_markdown(), idea_path)

        log_stage(workspace_dir, "hypothesis_construction",
                 f"Hypothesis construction completed. Generated {len(idea.candidates)} ideas, selected: {idea.selected_idea.content[:100]}...")

        return {
            "task": task,
            "idea": idea,
            "stage": "hypothesis_construction"
        }

    except Exception as e:
        log_stage(workspace_dir, "hypothesis_construction", f"Error: {str(e)}")
        raise WorkflowError(f"Hypothesis construction failed: {str(e)}")


def _parse_idea_from_debate(formatted_output: Dict[str, Any], debate_rounds: int) -> ResearchIdea:
    """Parse FormatterAgent output into ResearchIdea object"""

    candidates = []

    ideas_data = formatted_output.get("ideas", [])

    for idx, idea_data in enumerate(ideas_data):
        candidate = IdeaCandidate(
            content=idea_data.get("content", ""),
            score=idea_data.get("score", 0.0),
            round=idea_data.get("round", 0),
            criticisms=idea_data.get("weaknesses", [])
        )
        candidates.append(candidate)

    research_idea = ResearchIdea(
        candidates=candidates,
        selected_index=0,  # First one is best (already ranked by FormatterAgent)
        debate_rounds=debate_rounds
    )

    return research_idea
