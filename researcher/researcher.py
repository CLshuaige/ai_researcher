from pathlib import Path
from typing import Optional
import time

from researcher.config import config, ResearcherConfig
from researcher.state import ResearchState
from researcher.graph.researcher_graph import build_researcher_graph
from researcher.utils import initialize_workspace, save_markdown, load_markdown, get_artifact_path
from researcher.schemas import ResearchIdea, ExperimentalMethod, ExperimentResult, ReviewReport


class AIResearcher:
    """
    Main class for AI-powered research automation system

    Manages the complete research workflow from task definition to paper review.
    Uses LangGraph for workflow orchestration and AG2 for multi-agent collaboration.

    Args:
        project_name: Name of the research project
        workspace_dir: Optional custom workspace directory
        clear_workspace: Whether to clear existing workspace on initialization
        model_preset: Optional model preset name from registry
            (e.g., "openai-gpt4o", "qwen-30b-local"). If None, uses the
            default preset from configuration.
    """

    def __init__(
        self,
        project_name: str = "research_project",
        workspace_dir: Optional[Path] = None,
        clear_workspace: bool = False,
        model_preset: Optional[str] = None
    ):
        self.project_name = project_name

        # If a model preset is provided, update the global configuration model
        if model_preset is not None:
            from researcher.config import config as global_config
            from researcher.llm import get_model_preset
            model_config = get_model_preset(model_preset)
            global_config.model = model_config
            print(f"Using model preset: {model_preset} ({model_config.provider}/{model_config.model_name})")

        # Setup workspace
        if workspace_dir:
            self.workspace_dir = Path(workspace_dir)
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.workspace_dir = config.workspace.get_project_dir(project_name)

        if clear_workspace:
            import shutil
            if self.workspace_dir.exists():
                shutil.rmtree(self.workspace_dir)
            self.workspace_dir.mkdir(parents=True, exist_ok=True)

        initialize_workspace(self.workspace_dir)

        self.graph = build_researcher_graph()
        self.current_state: Optional[ResearchState] = None

    def run(self, input_text: str, input_file: Optional[Path] = None) -> ResearchState:
        """
        Execute the complete research workflow

        Args:
            input_text: Initial research prompt or task description
            input_file: Optional path to input.md file

        Returns:
            Final research state with all artifacts
        """
        start_time = time.time()

        if input_file and input_file.exists():
            with open(input_file, 'r', encoding='utf-8') as f:
                input_text = f.read()

        input_path = get_artifact_path(self.workspace_dir, "input")
        save_markdown(input_text, input_path)

        initial_state: ResearchState = {
            "input_text": input_text,
            "task": None,
            "literature": None,
            "idea": None,
            "method": None,
            "results": None,
            "paper": None,
            "referee": None,
            "current_round": 0,
            "debate_history": [],
            "workspace_dir": self.workspace_dir,
            "project_name": self.project_name,
            "stage": "initialization",
            "error": None
        }

        print(f"Starting research workflow for project: {self.project_name}")
        print(f"Workspace: {self.workspace_dir}")

        final_state = self.graph.invoke(initial_state)

        # Store state
        self.current_state = final_state

        # Report completion
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"\nWorkflow completed in {minutes} min {seconds} sec")
        print(f"Final stage: {final_state['stage']}")

        return final_state

    # Getters
    def get_workspace_path(self) -> Path:
        """Get the workspace directory path"""
        return self.workspace_dir

    def get_artifact(self, artifact_name: str) -> Optional[str]:
        """Retrieve specific artifact content as string"""
        artifact_path = get_artifact_path(self.workspace_dir, artifact_name)
        return load_markdown(artifact_path)

    def get_idea(self) -> Optional[ResearchIdea]:
        """Get the research idea object"""
        return self.current_state["idea"] if self.current_state else None

    def get_method(self) -> Optional[ExperimentalMethod]:
        """Get the experimental method object"""
        return self.current_state["method"] if self.current_state else None

    def get_results(self) -> Optional[ExperimentResult]:
        """Get the experiment results object"""
        return self.current_state["results"] if self.current_state else None

    def get_referee(self) -> Optional[ReviewReport]:
        """Get the review report object"""
        return self.current_state["referee"] if self.current_state else None

    # Display methods
    def show_idea(self) -> None:
        """Display the research idea"""
        idea = self.get_idea()
        if idea:
            print(idea.to_markdown())
        else:
            print("No idea generated yet")

    def show_method(self) -> None:
        """Display the experimental method"""
        method = self.get_method()
        if method:
            print(method.to_markdown())
        else:
            print("No method generated yet")

    def show_results(self) -> None:
        """Display the experiment results"""
        results = self.get_results()
        if results:
            print(results.to_markdown())
        else:
            print("No results generated yet")

    def show_referee(self) -> None:
        """Display the review report"""
        referee = self.get_referee()
        if referee:
            print(referee.to_markdown())
        else:
            print("No review generated yet")

    def show_workspace(self) -> None:
        """Display workspace structure"""
        print(f"\nWorkspace: {self.workspace_dir}")
        print("\nGenerated artifacts:")
        for artifact in ["task", "literature", "idea", "method", "results", "paper", "referee"]:
            path = get_artifact_path(self.workspace_dir, artifact)
            if path.exists():
                print(f"  ✓ {artifact}: {path}")
            else:
                print(f"  ✗ {artifact}: not generated")
