from typing import Dict, Any, List, Annotated
from pathlib import Path

from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group import (
    AgentTarget,
    FunctionTarget,
    FunctionTargetResult,
    RevertToUserTarget,
)

from researcher.state import ResearchState
from researcher.schemas import LiteratureReview, LiteratureItem
from researcher.agents import LiteratureSearcherAgent, LiteratureSummarizerAgent
from researcher.utils import (
    save_markdown,
    save_json,
    log_stage,
    get_artifact_path,
    load_artifact_from_file,
    get_llm_config,
    save_agent_history,
)
from researcher.prompts.templates import LITERATURE_SEARCH_PROMPT, LITERATURE_SUMMARY_PROMPT
from researcher.exceptions import WorkflowError


def literature_review_node(state: ResearchState) -> Dict[str, Any]:
    """Conduct literature review"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "literature_review", "Starting literature review")

    try:
        task = load_artifact_from_file(workspace_dir, "task")
        if not task:
            raise WorkflowError("Task file not found")

        config = state["config"]["researcher"]["literature_review"]
        num_papers = config["num_papers"]

        llm_config = get_llm_config()

        # Define arxiv search tool
        def search_arxiv_papers(
            query: Annotated[str, "Search query for arxiv academic database"],
            max_results: Annotated[int, "Maximum number of papers to retrieve"] = 5
        ) -> dict:
            """Search arxiv for academic papers, download and cache PDFs, return paper metadata"""
            try:
                import arxiv
                from datetime import datetime

                cache_dir = workspace_dir / "literature" / "arxiv_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                search = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.Relevance
                )

                papers = []
                papers_text_parts = []
                cache_metadata = {
                    "timestamp": timestamp,
                    "query": query,
                    "max_results": max_results,
                    "papers": []
                }

                for result in search.results():
                    arxiv_id = result.entry_id.split('/')[-1]
                    paper_data = {
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "abstract": result.summary,
                        "url": result.entry_id,
                        "year": result.published.year if result.published else None,
                        "arxiv_id": arxiv_id
                    }
                    papers.append(paper_data)

                    papers_text_parts.append(
                        f"Title: {result.title}\n"
                        f"Authors: {', '.join([a.name for a in result.authors])}\n"
                        f"Abstract: {result.summary}\n"
                    )

                    # Cache PDF
                    pdf_filename = None
                    try:
                        pdf_filename = f"{timestamp}_{arxiv_id}.pdf"
                        result.download_pdf(dirpath=str(cache_dir), filename=pdf_filename)
                        paper_data["pdf_cached"] = pdf_filename
                    except Exception:
                        paper_data["pdf_cached"] = None

                    cache_metadata["papers"].append({
                        "arxiv_id": arxiv_id,
                        "title": result.title,
                        "query": query,
                        "pdf_file": pdf_filename,
                        "url": result.entry_id
                    })

                if cache_metadata["papers"]:
                    metadata_path = cache_dir / f"{timestamp}_metadata.json"
                    save_json(cache_metadata, metadata_path)

                # Format papers text
                papers_text = "\n\n---\n\n".join(papers_text_parts) if papers_text_parts else "No papers found"

                return {
                    "success": True,
                    "papers": papers,
                    "query": query,
                    "formatted_text": papers_text
                }

            except ImportError:
                return {"success": False, "error": "arxiv package not installed"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        searcher = LiteratureSearcherAgent().create_agent(llm_config, functions=[search_arxiv_papers])
        summarizer = LiteratureSummarizerAgent().create_agent(llm_config)

        pattern = DefaultPattern(
            initial_agent=searcher,
            agents=[searcher, summarizer],
            group_manager_args={"llm_config": llm_config}
        )

        def format_papers_for_summarizer(context_variables):
            message = f"""{LITERATURE_SUMMARY_PROMPT.format(
                papers="Please extract and synthesize the papers from the search results above (look for formatted_text in the tool results).",
                task=task
            )}"""
            
            return FunctionTargetResult(
                target=AgentTarget(summarizer),
                message=message,
                context_variables=context_variables
            )

        searcher.handoffs.set_after_work(FunctionTarget(format_papers_for_summarizer))
        summarizer.handoffs.set_after_work(RevertToUserTarget())

        prompt = LITERATURE_SEARCH_PROMPT.format(task=task)

        result, context, last_agent = initiate_group_chat(
            pattern=pattern,
            messages=prompt,
            max_rounds=10
        )

        # Extract papers and synthesis from messages
        literature_items = []
        synthesis = None

        for msg in result.messages:
            content = msg.get("content", "")

            if "arxiv_id" in content and isinstance(content, str):
                try:
                    import json
                    data = json.loads(content) if content.startswith("{") else None
                    if data and data.get("success") and "papers" in data:
                        for paper in data["papers"]:
                            literature_items.append(LiteratureItem(
                                title=paper.get("title", ""),
                                authors=paper.get("authors", []),
                                abstract=paper.get("abstract", ""),
                                url=paper.get("url", ""),
                                year=paper.get("year")
                            ))
                except:
                    pass

            if msg.get("name") == summarizer.name and len(content) > 100:
                synthesis = content

        if not synthesis:
            raise WorkflowError("Summarizer did not generate synthesis")

        save_agent_history(
            workspace_dir=workspace_dir,
            node_name="literature_review",
            messages=result.messages,
            agent_chat_messages={
                searcher.name: searcher.chat_messages,
                summarizer.name: summarizer.chat_messages
            }
        )

        literature = LiteratureReview(items=literature_items, synthesis=synthesis)

        lit_dir = workspace_dir / "literature"
        lit_dir.mkdir(exist_ok=True)

        lit_path = get_artifact_path(workspace_dir, "literature")
        save_markdown(literature.to_markdown(), lit_path)

        papers_json_path = lit_dir / "papers.json"
        save_json([item.model_dump() for item in literature_items], papers_json_path)

        log_stage(workspace_dir, "literature_review", f"Completed. Found {len(literature_items)} papers")

        return {"task": task, "literature": literature, "stage": "literature_review"}

    except Exception as e:
        log_stage(workspace_dir, "literature_review", f"Error: {str(e)}")
        raise WorkflowError(f"Literature review failed: {str(e)}")
