from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Annotated
import json
import os
import re
import shutil

from autogen import ConversableAgent
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group import (
    AgentTarget,
    FunctionTarget,
    FunctionTargetResult,
    TerminateTarget,
    ContextVariables,
    ReplyResult,
)

from researcher.state import ResearchState
from researcher.schemas import LiteratureReview, LiteratureItem
from researcher.agents import LiteratureSearcherAgent, LiteratureSummarizerAgent, LiteratureManagerAgent
from researcher.utils import (
    save_markdown,
    save_json,
    load_json,
    log_stage,
    get_artifact_path,
    load_artifact_from_file,
    load_markdown,
    get_llm_config,
    save_agent_history,
    iterable_group_chat,
    parse_json_from_response,
    markdown_to_pdf,
    latex_to_pdf,
    get_relative_path,
)
from researcher.prompts.templates import (
    LITERATURE_MANAGER_INITIAL_PROMPT,
    LITERATRUE_SEARCH_PROMPT_WITH_MANAGER,
    LITERATURE_SEARCH_PROMPT,
    LITERATURE_SUMMARY_PROMPT,
    LITERATURE_BLOG_PROMPT,
    LITERATURE_SYNTHESIS_FROM_BLOGS_PROMPT,
    LITERATURE_SYNTHESIS_FROM_BLOGS_PROMPT_LATEX
)
from researcher.exceptions import WorkflowError
from researcher.integrations.literature_search import search_literature


def literature_review_node(state: ResearchState) -> Dict[str, Any]:
    """Conduct literature review"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "literature_review", "Starting literature review")

    try:
        task = load_artifact_from_file(workspace_dir, "task")
        if not task:
            raise WorkflowError("Task file not found")

        config = state["config"]["researcher"]["literature_review"]
        mode = config["mode"]
        use_manager = config.get("use_manager", False)
        num_papers = config["num_papers"]
        format = config.get("output_format", "markdown").lower()
        sources = [s.lower() for s in config.get("sources", ["arxiv"])]
        api_config = config.get("api") or {}
        test_summary = False

        llm_config = get_llm_config()

        def search_literature_papers(
            context_variables: ContextVariables,
            query: Annotated[str, "Search query for academic databases"],
            max_results: Annotated[int, "Maximum number of papers to retrieve per source"] = num_papers,
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
                import json
                results_dict = results[0]
                if results_dict["success"]:
                    formatted_text = results_dict.get("formatted_text", "")
                    formatted_results = f"Search results for '{query}':\n{formatted_text}\n\n Searching completed."
                else:
                    formatted_results = f"No papers found. Please try again."

                return ReplyResult(
                    message=formatted_results,
                    context_variables=context_variables
                )

            except Exception as e:
                return {"success": False, "error": str(e)}

        if use_manager:
            manager = LiteratureManagerAgent().create_agent(llm_config)
        searcher = LiteratureSearcherAgent().create_agent(
            llm_config,
            functions=[search_literature_papers],
        )
        summarizer = LiteratureSummarizerAgent().create_agent(llm_config, enable_context_compression=False)

        def build_paper_blog(entry: Dict[str, Any]) -> str:
            parsed_md_path = _resolve_workspace_path(entry["parsed_md_path"], workspace_dir)
            paper_markdown = load_markdown(parsed_md_path) or ""
            images_text = (
                "The parsed paper markdown above already contains the extracted figures at their insertion points. "
                "Choose important figures from that markdown context."
            )

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

            blogger = ConversableAgent(
                name="Blogger",
                human_input_mode="NEVER",
                system_message=(
                    "Read one paper and write a high-quality markdown blog. "
                    "Keep academic readability and focus on motivation, method, evidence, limitations, relevance. "
                    "Only keep the most important figures."
                ),
                llm_config=llm_config,
            )
            reply = blogger.generate_reply(
                messages=[{"role": "user", "content": blog_prompt}],
                sender=None,
            )

            content = str(reply["content"]).strip()
            content = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n", "", content)
            content = re.sub(r"\n?```$", "", content).strip()
            return content

        def prepare_summary_input(output: Any, context_variables: ContextVariables):
            metadata_path = workspace_dir / "literature" / "metadata.json"
            metadata_path = Path(metadata_path)
            unified_metadata: List[Dict[str, Any]] = []
            if not test_summary:
                searching_results = context_variables.get("searching_results", [])
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

                        if mode == "detailed":
                            entry.update(_prepare_paper_bundle(entry, workspace_dir))
                        elif mode == "basic":
                            entry.update({
                                "parse_status": "basic_mode",
                                "parser_used": "none",
                                "paper_bundle_dir": "",
                                "bundle_pdf_path": "",
                                "parsed_md_path": "",
                                "images_dir": "",
                                "blog_path": "",
                            })

                        unified_metadata.append(entry)

                save_json(
                    {
                        "generated_at": datetime.now().isoformat(),
                        "task": task,
                        "papers": unified_metadata,
                    },
                    metadata_path,
                )

            indexed_blogs: Dict[int, str] = {}
            unified_metadata = load_json(metadata_path)["papers"]
            max_workers = len(unified_metadata)
            # for blogs
            if mode == "detailed":
                blog_blocks: List[str] = []
                if not test_summary:
                    # build blogs in parallel
                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        futures = {
                            pool.submit(build_paper_blog, entry): idx
                            for idx, entry in enumerate(unified_metadata)
                        }
                        # wait for all blogs to be built
                        from tqdm import tqdm
                        for future in tqdm(as_completed(futures), total=len(futures), desc="Building blogs"):
                            idx = futures[future]
                            indexed_blogs[idx] = future.result()

                    # assemble blog blocks
                    for idx, entry in enumerate(unified_metadata, start=1):
                        blog_content = indexed_blogs[idx - 1]
                        blog_path = _resolve_workspace_path(entry["blog_path"], workspace_dir)
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
                        markdown_to_pdf(blog_path)

                        citation_hint = f"{', '.join(entry.get('authors', [])[:2])} ({entry.get('year')})"
                        blog_content_for_summary = _rewrite_markdown_image_paths(
                            blog_content,
                            source_dir=blog_path.parent,
                            target_dir=workspace_dir,
                        )
                        blog_blocks.append(
                            f"## Paper {idx}\n"
                            f"- Citation Hint: {citation_hint}\n"
                            "\n"
                            f"{blog_header}{blog_content_for_summary}"
                        )

                else:
                    # load blogs from file
                    for idx, entry in enumerate(unified_metadata, start=1):
                        blog_path = _resolve_workspace_path(entry["blog_path"], workspace_dir)
                        full_blog_content = load_markdown(blog_path) or ""

                        citation_hint = f"{', '.join(entry.get('authors', [])[:2])} ({entry.get('year')})"
                        blog_content_for_summary = _rewrite_markdown_image_paths(
                            full_blog_content,
                            source_dir=blog_path.parent,
                            target_dir=workspace_dir,
                        )
                        blog_blocks.append(
                            f"## Paper {idx}\n"
                            f"- Citation Hint: {citation_hint}\n"
                            "\n"
                            f"{blog_content_for_summary}"
                        )

                if format == "latex":
                    template_path = "/home/ai_researcher/projects/ai_researcher/researcher/latex/literature/template.tex"
                    template_path = Path(template_path)
                    template_content = load_markdown(template_path)
                    summary_prompt = LITERATURE_SYNTHESIS_FROM_BLOGS_PROMPT_LATEX.format(
                        task=task,
                        blogs_text="\n\n---\n\n".join(blog_blocks),
                        template=template_content,
                    )
                elif format == "markdown":
                    summary_prompt = LITERATURE_SYNTHESIS_FROM_BLOGS_PROMPT.format(
                        task=task,
                        blogs_text="\n\n---\n\n".join(blog_blocks),
                    )
            else:
                abstract_blocks: List[str] = []
                for idx, entry in enumerate(unified_metadata, start=1):
                    abstract_blocks.append(
                        f"Paper {idx}\n"
                        f"Title: {entry.get('title', '')}\n"
                        f"Authors: {', '.join(entry.get('authors', []))}\n"
                        f"Year: {entry.get('year')}\n"
                        f"Source: {entry.get('source', '')}\n"
                        f"URL: {entry.get('url', '')}\n"
                        f"Abstract: {entry.get('abstract', '')}\n"
                    )

                if format == "latex":
                    template_path = "/home/ai_researcher/projects/ai_researcher/researcher/latex/literature/template.tex"
                    template_path = Path(template_path)
                    template_content = load_markdown(template_path)
                    from string import Template
                    prompt_template = Template(LITERATURE_SYNTHESIS_FROM_BLOGS_PROMPT_LATEX)
                    summary_prompt = prompt_template.substitute(
                        task=task,
                        blogs_text="\n\n---\n\n".join(abstract_blocks),
                        template=template_content,
                    )
                elif format == "markdown":
                    summary_prompt = LITERATURE_SYNTHESIS_FROM_BLOGS_PROMPT.format(
                        task=task,
                        blogs_text="\n\n---\n\n".join(abstract_blocks),
                    )

            context_variables["unified_metadata"] = unified_metadata
            return summary_prompt, context_variables

        def route_to_next_agent(output: Any, context_variables: ContextVariables) -> FunctionTargetResult:
            state = context_variables.get("state", "start")
            if state == "searched":
                decision = str(output)
                if "SEARCH_COMPLETE" in decision or not use_manager:
                    context_variables["state"] = "completed"
                    summary_prompt, context_variables = prepare_summary_input(output, context_variables)
                    return FunctionTargetResult(
                        target=AgentTarget(summarizer),
                        messages=summary_prompt,
                        context_variables=context_variables,
                    )
            
            context_variables["state"] = "searching"
            return FunctionTargetResult(
                target=AgentTarget(searcher),
                messages=LITERATRUE_SEARCH_PROMPT_WITH_MANAGER,
                context_variables=context_variables,
            )
        
        def manage_history_messages(messages: list[dict]) -> list[dict]:
            """
            Only keep the system message and the last message for the summarizer.
            """
            return [messages[-1]] # Last message
        
        ctx = ContextVariables(data={
            "state": "start",
            "task": task,
            "searching_results": [],
        })

        if use_manager:
            pattern = DefaultPattern(
                initial_agent=manager,
                agents=[searcher, summarizer, manager],
                group_manager_args={"llm_config": llm_config},
                context_variables=ctx,
            )
        elif test_summary:
            pattern = DefaultPattern(
                initial_agent=summarizer,
                agents=[summarizer],
                group_manager_args={"llm_config": llm_config},
                context_variables=ctx,
            )
        else:
            pattern = DefaultPattern(
                initial_agent=searcher,
                agents=[searcher, summarizer],
                group_manager_args={"llm_config": llm_config},
                context_variables=ctx,
            )

        if use_manager:
            manager.handoffs.set_after_work(FunctionTarget(route_to_next_agent))
            searcher.handoffs.set_after_work(AgentTarget(manager))
        else:
            searcher.handoffs.set_after_work(FunctionTarget(route_to_next_agent))
        summarizer.register_hook(
            "process_all_messages_before_reply",
            manage_history_messages
        )
        summarizer.handoffs.set_after_work(TerminateTarget())

        if use_manager:
            prompt = LITERATURE_MANAGER_INITIAL_PROMPT.format(task=task)
        elif test_summary:
            prompt = prepare_summary_input(None, ctx)[0]
        else:
            prompt = LITERATURE_SEARCH_PROMPT.format(task=task, num_papers=num_papers)


        if state["config"]["researcher"]["iterable"]:
            result, context, _ = iterable_group_chat(
                state,
                max_rounds=20,
                enable_hitl=False,
                pattern=pattern,
                prompt=prompt,
            )
        else:
            result, context, _ = initiate_group_chat(
                pattern=pattern,
                messages=prompt,
                max_rounds=20,
            )

        unified_metadata = context.get("unified_metadata", [])

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

        synthesis = None
        for msg in reversed(result.chat_history):
            content = msg.get("content", "")
            if msg.get("name") == summarizer.name and isinstance(content, str) and content.strip():
                synthesis = content
                break

        if not synthesis:
            raise WorkflowError("Summarizer did not generate synthesis")

        save_agent_history(
            workspace_dir=workspace_dir,
            node_name="literature_review",
            messages=result.chat_history,
            agent_chat_messages={
                searcher.name: searcher.chat_messages,
                summarizer.name: summarizer.chat_messages
            }
        )

        literature = LiteratureReview(items=literature_items, synthesis=synthesis)

        lit_dir = workspace_dir / "literature"
        lit_dir.mkdir(exist_ok=True)

        lit_path = get_artifact_path(workspace_dir, "literature")
        if format == "latex":
            lit_path = lit_path.with_suffix(".tex")
            latex_content = literature.to_latex()
            save_markdown(latex_content, lit_path)
            latex_to_pdf(lit_path)
        elif format == "markdown":
            save_markdown(literature.to_markdown(), lit_path)
            markdown_to_pdf(lit_path)

        log_stage(workspace_dir, "literature_review", f"Completed. Found {len(literature_items)} papers")

        update_state = {
            "metadata": literature_items,
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
        candidate = _resolve_workspace_path(entry["pdf_path"], workspace_dir)
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
            "paper_bundle_dir": get_relative_path(bundle_dir, workspace_dir),
            "bundle_pdf_path": "",
            "parsed_md_path": get_relative_path(parsed_md_path, workspace_dir),
            "images_dir": get_relative_path(images_dir, workspace_dir),
            "blog_path": get_relative_path(bundle_dir / "blog.md", workspace_dir),
        }

    bundle_dir = pdf_path.parent / pdf_path.stem
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_pdf_path = bundle_dir / "paper.pdf"
    if pdf_path != bundle_pdf_path:
        if bundle_pdf_path.exists():
            if pdf_path.exists():
                pdf_path.unlink()
        else:
            shutil.move(str(pdf_path), str(bundle_pdf_path))

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
                pixmap = fitz.Pixmap(doc, xref)
                if pixmap.n >= 5:
                    pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
                image_name = f"page_{page_idx:03d}_img_{image_idx:02d}.png"
                image_path = images_dir / image_name
                pixmap.save(str(image_path))
                pixmap = None
                image_count += 1
                image_rel_path = get_relative_path(image_path, parsed_md_path.parent)
                chunk_lines.append(f"![Figure p{page_idx}-{image_idx}]({image_rel_path})")

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
        "pdf_path": get_relative_path(bundle_pdf_path, workspace_dir),
        "paper_bundle_dir": get_relative_path(bundle_dir, workspace_dir),
        "bundle_pdf_path": get_relative_path(bundle_pdf_path, workspace_dir),
        "parsed_md_path": get_relative_path(parsed_md_path, workspace_dir),
        "images_dir": get_relative_path(images_dir, workspace_dir),
        "blog_path": get_relative_path(bundle_dir / "blog.md", workspace_dir),
    }


def _resolve_workspace_path(path_value: Any, workspace_dir: Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else workspace_dir / path


def _rewrite_markdown_image_paths(markdown_text: str, source_dir: Path, target_dir: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        alt = match.group(1)
        raw_path = match.group(2).strip()
        if re.match(r"^(?:https?:|data:|file:|#)", raw_path):
            return match.group(0)
        resolved = (source_dir / raw_path).resolve()
        relative_path = Path(os.path.relpath(resolved, start=target_dir.resolve())).as_posix()
        return f"![{alt}]({relative_path})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace, markdown_text)
