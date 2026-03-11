import argparse
from pathlib import Path

from researcher.researcher import AIResearcher
from researcher.utils import load_global_config

from researcher.api.app import app

def main():
    parser = argparse.ArgumentParser(
        description="AI Researcher: Automated scientific research system"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Research task description or path to input.md file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/debug.yaml",
        help="Path to configuration YAML file, which decide the researcher behavior"

    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="research_project",
        help="Project name for workspace organization"
    )
    parser.add_argument(
        "--workspace-dir", 
        type=str, 
        default=None,
        help="Path to workspace directory"
    )
    parser.add_argument(
        "--clear-workspace",
        action="store_true",
        help="Clear workspace directory before starting"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model preset name from registry (e.g., 'openai-gpt4o', 'qwen-local'). "
             "Use '--list-models' to see available presets."
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available model presets and exit"
    )
    parser.add_argument(
        "--start-node",
        type=str,
        default=None,
        help="The task node to start the research workflow, choices: ['task_parsing', 'literature_review', 'hypothesis_construction', 'method_design', 'experiment_execution', ...]"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "step"],
        default=None,
        help="Run mode: auto (full workflow) or step (single node)"
    )

    args = parser.parse_args()

    # If need to list models, show and exit
    if args.list_models:
        from researcher.llm import list_available_models
        models = list_available_models()
        print("\nAvailable model presets:")
        print("=" * 60)
        for name, config in models.items():
            print(f"  {name:30s} | {config.provider:10s} | {config.model_name}")
            if config.base_url:
                print(f"    {'':30s} | base_url: {config.base_url}")
        print("=" * 60)
        return

    # Determine if input is a file path or text
    input_path = Path(args.input) if args.input else None
    if input_path and input_path.exists():
        input_text = None
        input_file = input_path
    else:
        input_text = args.input or "Define a novel research problem in AI."
        input_file = None

    # Load configuration YAML
    config = load_global_config(Path(args.config))

    # Initialize and run researcher
    researcher = AIResearcher(
        project_name=args.project_name,
        workspace_dir=args.workspace_dir,
        clear_workspace=args.clear_workspace,
        model_preset=args.model
    )
    print(f"Workspace: {researcher.get_workspace_path()}")
    print("Starting research workflow...")

    final_state = researcher.run(
        input_text=input_text,
        input_file=input_file,
        start_node=args.start_node,
        config=config,
        mode=args.mode,
    )

    print(f"\nWorkflow completed. Stage: {final_state['stage']}")
    print(f"Workspace: {researcher.get_workspace_path()}")


if __name__ == "__main__":
    main()
