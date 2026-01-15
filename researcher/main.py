import argparse
from pathlib import Path

from researcher.researcher import AIResearcher


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
        "--project-name",
        type=str,
        default="research_project",
        help="Project name for workspace organization"
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
        input_text = args.input or "Default research task"
        input_file = None

    # Initialize and run researcher
    researcher = AIResearcher(
        project_name=args.project_name,
        model_preset=args.model
    )
    print(f"Workspace: {researcher.get_workspace_path()}")
    print("Starting research workflow...")

    final_state = researcher.run(input_text=input_text, input_file=input_file)

    print(f"\nWorkflow completed. Stage: {final_state['stage']}")
    print(f"Workspace: {researcher.get_workspace_path()}")


if __name__ == "__main__":
    main()
