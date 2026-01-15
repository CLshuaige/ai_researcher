from typing import Dict, Any, List
import json

from researcher.state import ResearchState
from researcher.schemas import LiteratureReview, LiteratureItem
from researcher.agents import SearcherAgent, SummarizerAgent
from researcher.config import config
from researcher.utils import save_markdown, save_json, log_stage, get_artifact_path
from researcher.prompts.templates import LITERATURE_SEARCH_PROMPT, LITERATURE_SUMMARY_PROMPT
from researcher.llm import get_llm_client
from researcher.exceptions import WorkflowError


def literature_review_node(state: ResearchState) -> Dict[str, Any]:
    """Conduct literature review through sequential agent calls"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "literature_review", "Starting literature review")

    try:
        searcher = SearcherAgent()
        summarizer = SummarizerAgent()
        llm_client = get_llm_client(config.model)

        search_prompt = LITERATURE_SEARCH_PROMPT.format(task=state["task"])
        search_messages = [
            {"role": "system", "content": searcher.system_prompt},
            {"role": "user", "content": search_prompt}
        ]

        log_stage(workspace_dir, "literature_review", "Generating search keywords")
        search_response = llm_client.generate(search_messages)

        keywords_data = _parse_json_response(search_response)
        keywords = keywords_data.get("keywords", [])
        queries = keywords_data.get("queries", [])

        log_stage(workspace_dir, "literature_review", f"Generated {len(keywords)} keywords, {len(queries)} queries")

        literature_items, papers_text = _fetch_papers(queries, keywords, workspace_dir)

        summary_prompt = LITERATURE_SUMMARY_PROMPT.format(
            papers=papers_text,
            task=state["task"]
        )
        summary_messages = [
            {"role": "system", "content": summarizer.system_prompt},
            {"role": "user", "content": summary_prompt}
        ]

        log_stage(workspace_dir, "literature_review", "Synthesizing literature")
        synthesis = llm_client.generate(summary_messages)

        literature = LiteratureReview(
            items=literature_items,
            synthesis=synthesis
        )

        lit_dir = workspace_dir / "literature"
        lit_dir.mkdir(exist_ok=True)

        lit_path = get_artifact_path(workspace_dir, "literature")
        save_markdown(literature.to_markdown(), lit_path)

        papers_json_path = lit_dir / "papers.json"
        save_json([item.model_dump() for item in literature_items], papers_json_path)

        log_stage(workspace_dir, "literature_review", f"Literature review completed. Found {len(literature_items)} papers")

        return {
            "literature": literature,
            "stage": "literature_review"
        }

    except Exception as e:
        log_stage(workspace_dir, "literature_review", f"Error: {str(e)}")
        raise WorkflowError(f"Literature review failed: {str(e)}")


def _parse_json_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response"""
    try:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


def _fetch_papers(queries: List[str], keywords: List[str], workspace_dir=None) -> tuple[List[LiteratureItem], str]:
    """Fetch papers from academic sources and cache them"""
    literature_items = []
    papers_text_parts = []

    try:
        import arxiv
        from datetime import datetime

        cache_dir = workspace_dir / "literature" / "arxiv_cache" if workspace_dir else None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_metadata = {
            "timestamp": timestamp,
            "queries": queries[:3],
            "keywords": keywords,
            "papers": []
        }

        for query in queries[:3]:
            search = arxiv.Search(
                query=query,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )

            for result in search.results():
                arxiv_id = result.entry_id.split('/')[-1]

                item = LiteratureItem(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    url=result.entry_id,
                    year=result.published.year if result.published else None
                )
                literature_items.append(item)

                papers_text_parts.append(
                    f"Title: {result.title}\n"
                    f"Authors: {', '.join([a.name for a in result.authors])}\n"
                    f"Abstract: {result.summary}\n"
                )

                # Cache paper PDF
                if cache_dir:
                    try:
                        pdf_filename = f"{timestamp}_{arxiv_id}.pdf"
                        pdf_path = cache_dir / pdf_filename
                        result.download_pdf(dirpath=str(cache_dir), filename=pdf_filename)

                        cache_metadata["papers"].append({
                            "arxiv_id": arxiv_id,
                            "title": result.title,
                            "query": query,
                            "pdf_file": pdf_filename,
                            "url": result.entry_id
                        })
                    except Exception as e:
                        # Continue even if PDF download fails
                        pass

        # Save cache metadata
        if cache_dir and cache_metadata["papers"]:
            metadata_path = cache_dir / f"{timestamp}_metadata.json"
            save_json(cache_metadata, metadata_path)

        papers_text = "\n\n---\n\n".join(papers_text_parts) if papers_text_parts else "No papers found"

    except ImportError:
        papers_text = "[TODO: Install arxiv package for literature search]\n\nPlaceholder paper content for synthesis."
    except Exception as e:
        papers_text = f"[Error fetching papers: {str(e)}]\n\nPlaceholder paper content for synthesis."

    return literature_items, papers_text
