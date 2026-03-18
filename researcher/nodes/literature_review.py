from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List
import json
import re
import shutil

from autogen import ContextExpression, ConversableAgent
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import (
    AgentTarget,
    ContextVariables,
    ExpressionContextCondition,
    OnContextCondition,
    ReplyResult,
    TerminateTarget,
)
from autogen.agentchat.group.patterns import DefaultPattern

from researcher.agents import LiteratureBloggerAgent, LiteratureSearcherAgent, LiteratureSummarizerAgent
from researcher.exceptions import WorkflowError
from researcher.integrations.literature_search import search_literature
from researcher.prompts.templates import (
    LITERATURE_BLOG_PROMPT,
    LITERATURE_SEARCH_PROMPT,
    LITERATURE_SYNTHESIS_FROM_BLOGS_PROMPT,
)
from researcher.schemas import LiteratureItem, LiteratureReview
from researcher.state import ResearchState
from researcher.utils import (
    get_artifact_path,
    get_llm_config,
    iterable_group_chat,
    load_artifact_from_file,
    load_markdown,
    log_stage,
    save_agent_history,
    save_json,
    save_markdown,
)


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
        sources = [s.lower() for s in config.get("sources", ["arxiv"])]
        api_config = config.get("api") or {}

        llm_config = get_llm_config()

        def search_literature_papers(
            context_variables: ContextVariables,
            query: Annotated[str, "Search query for academic databases"],
            max_results: Annotated[int, "Maximum number of papers to retrieve per source"] = 5,
        ) -> dict:
            try:
                results = search_literature(
                    query=query,
                    max_results=max_results,
                    sources=sources,
                    workspace_dir=workspace_dir,
                    api_config=api_config,
                )

                context_variables["state"] = "searched"
                context_variables["searching_results"].extend(results)

                return ReplyResult(
                    message="Searching complete.",
                    context_variables=context_variables
                )

            except Exception as e:
                return {"success": False, "error": str(e)}

        searcher = LiteratureSearcherAgent().create_agent(llm_config)
        summarizer = LiteratureSummarizerAgent().create_agent(llm_config)

        llm_config_tool = get_llm_config(use_tool=True)
        executor = ConversableAgent(
            name="SearchExecutor",
            human_input_mode="NEVER",
            system_message="Call the search tool according to received literature-search instructions.",
            llm_config=llm_config_tool,
            functions=[search_literature_papers],
        )

        initial_variables = ContextVariables(data={
            "state": "start",
            "task": task,
            "searching_results": []
        })

        search_pattern = DefaultPattern(
            initial_agent=searcher,
            agents=[searcher, executor],
            group_manager_args={"llm_config": llm_config},
            context_variables=initial_variables,
        )

        searcher.handoffs.set_after_work(AgentTarget(executor))
        executor.handoffs.add_context_condition(
            OnContextCondition(
                target=TerminateTarget(),
                condition=ExpressionContextCondition(ContextExpression("${state} == 'searched'")),
            )
        )

        prompt = LITERATURE_SEARCH_PROMPT.format(task=task)
        if state["config"]["researcher"]["iterable"]:
            search_result, context, _ = iterable_group_chat(
                state,
                max_rounds=10,
                enable_hitl=False,
                pattern=search_pattern,
                prompt=prompt,
            )
        else:
            search_result, context, _ = initiate_group_chat(
                pattern=search_pattern,
                messages=prompt,
                max_rounds=10,
            )

        searching_results = context.get("searching_results", [])
        if not searching_results:
            raise WorkflowError("No literature search results found")

        unified_metadata: List[Dict[str, Any]] = []
        sequence = 0
        for source_result in searching_results:
            source = str(source_result.get("source", "unknown"))
            query = str(source_result.get("query", ""))
            for paper in source_result.get("papers", []) or []:
                sequence += 1
                paper_uid = _build_paper_uid(source, sequence, paper)
                entry = {
                    "paper_uid": paper_uid,
                    "source": source,
                    "query": query,
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []) or [],
                    "abstract": paper.get("abstract", "") or "",
                    "url": paper.get("url", "") or "",
                    "year": paper.get("year"),
                    "arxiv_id": paper.get("arxiv_id"),
                    "external_id": paper.get("external_id"),
                    "doi": paper.get("doi"),
                    "pdf_cached": paper.get("pdf_cached"),
                    "pdf_path": paper.get("pdf_path"),
                }
                entry.update(_prepare_paper_bundle(entry, workspace_dir))
                unified_metadata.append(entry)

        metadata_path = workspace_dir / "literature" / "metadata.json"
        save_json(
            {
                "generated_at": datetime.now().isoformat(),
                "task": task,
                "papers": unified_metadata,
            },
            metadata_path,
        )

        blogger_histories: List[Dict[str, Any]] = []
        image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}

        for entry in unified_metadata:
            blogger = LiteratureBloggerAgent().create_agent(llm_config)
            parsed_md_path = Path(str(entry["parsed_md_path"]))
            paper_markdown = load_markdown(parsed_md_path) or ""

            images_dir = Path(str(entry["images_dir"]))
            image_paths: List[str] = []
            if images_dir.exists():
                image_paths = [
                    str(path)
                    for path in sorted(images_dir.iterdir())
                    if path.suffix.lower() in image_suffixes
                ]
            images_text = "\n".join(f"- {img}" for img in image_paths) if image_paths else "- (No images available)"

            paper_metadata = json.dumps(
                {
                    "paper_uid": entry.get("paper_uid"),
                    "title": entry.get("title"),
                    "authors": entry.get("authors", []),
                    "year": entry.get("year"),
                    "source": entry.get("source"),
                    "query": entry.get("query"),
                    "url": entry.get("url"),
                    "parse_status": entry.get("parse_status"),
                    "parsed_md_path": entry.get("parsed_md_path"),
                    "images_dir": entry.get("images_dir"),
                },
                ensure_ascii=False,
                indent=2,
            )

            blog_prompt = LITERATURE_BLOG_PROMPT.format(
                task=task,
                paper_metadata=paper_metadata,
                paper_markdown=paper_markdown,
                available_images=images_text,
            )
            blogger.handoffs.set_after_work(TerminateTarget())
            blogger_pattern = DefaultPattern(
                initial_agent=blogger,
                agents=[blogger],
                group_manager_args={"llm_config": llm_config},
            )
            blogger_result, _, _ = initiate_group_chat(
                pattern=blogger_pattern,
                messages=blog_prompt,
                max_rounds=4,
            )
            blog_history = blogger_result.chat_history
            blogger_histories.extend(blog_history)
            blog_content = ""
            for msg in reversed(blog_history):
                if msg.get("name") == blogger.name and isinstance(msg.get("content"), str):
                    blog_content = msg.get("content", "")
                    if blog_content.strip():
                        break
            if not blog_content.strip():
                raise WorkflowError(f"Blogger did not generate content: {entry.get('title', '')}")

            blog_path = Path(str(entry["blog_path"]))
            blog_header = (
                "# Paper Metadata\n\n"
                f"- Title: {entry.get('title', '')}\n"
                f"- Authors: {', '.join(entry.get('authors', []))}\n"
                f"- Year: {entry.get('year')}\n"
                f"- Source: {entry.get('source', '')}\n"
                f"- URL: {entry.get('url', '')}\n"
                f"- Parse Status: {entry.get('parse_status', 'unknown')}\n\n"
                "---\n\n"
            )
            save_markdown(blog_header + blog_content, blog_path)

        blog_blocks: List[str] = []
        for idx, entry in enumerate(unified_metadata, start=1):
            blog_path = Path(str(entry["blog_path"]))
            blog_text = load_markdown(blog_path) or ""
            if not blog_text.strip():
                raise WorkflowError(f"Missing blog content for paper {idx}")

            citation_hint = f"{', '.join(entry.get('authors', [])[:2])} ({entry.get('year')})"
            blog_blocks.append(
                f"## Paper {idx}\n"
                f"- Citation Hint: {citation_hint}\n"
                "\n"
                f"{blog_text}"
            )
        blogs_text = "\n\n---\n\n".join(blog_blocks)

        summary_prompt = LITERATURE_SYNTHESIS_FROM_BLOGS_PROMPT.format(
            task=task,
            blogs_text=blogs_text,
        )
        summarizer.handoffs.set_after_work(TerminateTarget())
        summarizer_pattern = DefaultPattern(
            initial_agent=summarizer,
            agents=[summarizer],
            group_manager_args={"llm_config": llm_config},
        )
        summary_result, _, _ = initiate_group_chat(
            pattern=summarizer_pattern,
            messages=summary_prompt,
            max_rounds=5,
        )
        summary_history = summary_result.chat_history
        synthesis = ""
        for msg in reversed(summary_history):
            if msg.get("name") == summarizer.name and isinstance(msg.get("content"), str):
                synthesis = msg.get("content", "")
                if synthesis.strip():
                    break
        if not synthesis.strip():
            raise WorkflowError("Summarizer did not generate synthesis")

        literature_items = [
            LiteratureItem(
                title=entry.get("title", ""),
                authors=entry.get("authors", []),
                abstract=entry.get("abstract", ""),
                url=entry.get("url", ""),
                year=entry.get("year"),
            )
            for entry in unified_metadata
        ]
        literature = LiteratureReview(items=literature_items, synthesis=synthesis)

        lit_path = get_artifact_path(workspace_dir, "literature")
        save_markdown(literature.to_markdown(), lit_path)

        merged_history = list(search_result.chat_history) + blogger_histories + summary_history
        save_agent_history(
            workspace_dir=workspace_dir,
            node_name="literature_review",
            messages=merged_history,
            agent_chat_messages={
                searcher.name: searcher.chat_messages,
                summarizer.name: summarizer.chat_messages
            }
        )

        log_stage(workspace_dir, "literature_review", f"Completed. Found {len(literature_items)} papers")

        update_state = {
            "literature": literature,
            "stage": "literature_review",
        }

        # router
        if state["config"]["researcher"]["workflow"] == "default":
            update_state["next_node"] = "hypothesis_construction"
        return update_state

    except Exception as e:
        log_stage(workspace_dir, "literature_review", f"Error: {str(e)}")
        raise WorkflowError(f"Literature review failed: {str(e)}")


def _build_paper_uid(source: str, sequence: int, paper: Dict[str, Any]) -> str:
    raw = paper.get("arxiv_id") or paper.get("external_id") or paper.get("doi") or paper.get("url") or "paper"
    source_token = re.sub(r"[^a-zA-Z0-9._-]+", "_", source).strip("._")[:20] or "source"
    core = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(raw)).strip("._")[:96] or "paper"
    return f"{source_token}_{sequence:05d}_{core}"


def _prepare_paper_bundle(entry: Dict[str, Any], workspace_dir: Path) -> Dict[str, Any]:
    source = str(entry.get("source", "unknown"))
    pdf_path: Path | None = None

    if entry.get("pdf_path"):
        candidate = Path(str(entry["pdf_path"]))
        if candidate.exists():
            pdf_path = candidate

    if pdf_path is None and entry.get("pdf_cached"):
        candidate = workspace_dir / "literature" / f"{source}_cache" / str(entry["pdf_cached"])
        if candidate.exists():
            pdf_path = candidate

    if pdf_path is None:
        bundle_dir = workspace_dir / "literature" / "papers" / str(entry["paper_uid"])
        bundle_dir.mkdir(parents=True, exist_ok=True)
        images_dir = bundle_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        parsed_md_path = bundle_dir / "paper.md"
        save_markdown(
            "# Parsed Paper\n\n"
            "_Full-text PDF is unavailable. This is metadata-based content._\n\n"
            f"- Title: {entry.get('title', '')}\n"
            f"- Authors: {', '.join(entry.get('authors', []))}\n"
            f"- Year: {entry.get('year')}\n"
            f"- URL: {entry.get('url', '')}\n\n"
            "## Abstract\n\n"
            f"{entry.get('abstract', '')}\n",
            parsed_md_path,
        )
        return {
            "parse_status": "metadata_only",
            "parser_used": "none",
            "paper_bundle_dir": str(bundle_dir),
            "bundle_pdf_path": "",
            "parsed_md_path": str(parsed_md_path),
            "images_dir": str(images_dir),
            "blog_path": str(bundle_dir / "blog.md"),
        }

    bundle_dir = pdf_path.parent / pdf_path.stem
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_pdf_path = bundle_dir / "paper.pdf"
    if not bundle_pdf_path.exists():
        shutil.copy2(pdf_path, bundle_pdf_path)

    images_dir = bundle_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    parsed_md_path = bundle_dir / "paper.md"

    import fitz  # type: ignore

    try:
        doc = fitz.open(str(bundle_pdf_path))
        sections: List[str] = []
        image_count = 0

        for page_idx, page in enumerate(doc, start=1):
            text = (page.get_text("text") or "").strip()
            chunk_lines = [f"## Page {page_idx}", "", text if text else "_No text extracted on this page._"]

            for image_idx, image_info in enumerate(page.get_images(full=True), start=1):
                xref = image_info[0]
                image_data = doc.extract_image(xref)
                image_bytes = image_data.get("image")
                if not image_bytes:
                    continue
                ext = image_data.get("ext", "png")
                image_name = f"page_{page_idx:03d}_img_{image_idx:02d}.{ext}"
                image_path = images_dir / image_name
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)
                image_count += 1
                chunk_lines.append(f"![Figure p{page_idx}-{image_idx}]({image_path})")

            sections.append("\n".join(chunk_lines).strip())

        markdown_text = "\n\n".join(sections).strip()
        save_markdown(
            "# Parsed Paper\n\n"
            f"- Source PDF: {bundle_pdf_path}\n"
            "- Parser Used: fitz\n"
            f"- Extracted Images: {image_count}\n\n"
            f"{markdown_text}\n",
            parsed_md_path,
        )
        parse_status = "fulltext"
        parser_used = "fitz"
    except Exception:
        save_markdown(
            "# Parsed Paper\n\n"
            "_PDF parsing failed. Falling back to metadata-only content._\n\n"
            f"- Title: {entry.get('title', '')}\n"
            f"- Authors: {', '.join(entry.get('authors', []))}\n"
            f"- Year: {entry.get('year')}\n"
            f"- URL: {entry.get('url', '')}\n\n"
            "## Abstract\n\n"
            f"{entry.get('abstract', '')}\n",
            parsed_md_path,
        )
        parse_status = "metadata_only"
        parser_used = "fitz_failed"

    return {
        "parse_status": parse_status,
        "parser_used": parser_used,
        "paper_bundle_dir": str(bundle_dir),
        "bundle_pdf_path": str(bundle_pdf_path),
        "parsed_md_path": str(parsed_md_path),
        "images_dir": str(images_dir),
        "blog_path": str(bundle_dir / "blog.md"),
    }
