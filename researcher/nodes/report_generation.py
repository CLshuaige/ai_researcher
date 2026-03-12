from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re
import json
import shutil
import subprocess
import os
import requests

from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group import (
    ContextVariables,
    AgentTarget,
    FunctionTarget,
    FunctionTargetResult,
    TerminateTarget,
)
from autogen.agentchat.contrib.capabilities import transform_messages
from autogen.agentchat.contrib.capabilities.transforms import TextMessageCompressor

from researcher.state import ResearchState
from researcher.agents import PaperWriterAgent, OutlinerAgent, SectionWriterAgent
from researcher.utils import (
    log_stage,
    get_artifact_path,
    load_artifact_from_file,
    get_llm_config,
    parse_json_from_response,
    save_agent_history,
    save_markdown,
    iterable_group_chat
)
from researcher.prompts.paper_writing import (
    outline_prompt,
    keywords_prompt,
    abstract_initial_prompt,
    abstract_prompt,
    introduction_prompt,
    methods_prompt,
    results_prompt,
    conclusions_prompt,
    section_prompt,
    citation_addition_prompt,
    fix_latex_prompt,
)
from researcher.exceptions import WorkflowError
from researcher.latex.presets import Journal, get_preset


def report_generation_node(state: ResearchState) -> Dict[str, Any]:
    """Generate research paper with LaTeX structure"""
    workspace_dir = state["workspace_dir"]

    try:
        # Load configuration
        config = state["config"]["researcher"]["report_generation"]
        mode = config["mode"]
        skip_generation = config["skip_generation"]
        compiler_verification = config["compiler_verification"]
        output_type = config["output_type"]
        include_references = config["include_references"]
        citations_config = config["citations"]

        # Load artifacts
        task = load_artifact_from_file(workspace_dir, "task") or ""
        literature_md = load_artifact_from_file(workspace_dir, "literature") or ""
        idea_content = load_artifact_from_file(workspace_dir, "idea") or ""
        method_content = load_artifact_from_file(workspace_dir, "method") or ""
        results_content = load_artifact_from_file(workspace_dir, "results") or ""

        selected_idea = _extract_selected_idea(idea_content)
        bibliography_bib, cite_keys_info = _parse_literature_to_bibtex(workspace_dir)

        llm_config = get_llm_config()

        # Create agent with automatic context compression from global config
        paper_writer = PaperWriterAgent().create_agent(llm_config)
        outliner = OutlinerAgent().create_agent(llm_config)

        # If message is extremely long, here to consider to shorten it
        method_summary = method_content
        results_summary = results_content

        initial_context = ContextVariables(data={
            "writing_mode": "outline",  # outline -> abstract -> keywords -> section
            "current_section_index": 0,
            "outline_structure": None,
            "completed_sections": {},
            "paper_metadata": {},
            "bibliography_bib": bibliography_bib,
            "task": task,
            "idea": selected_idea,
            "method_summary": method_summary,
            "results_summary": results_summary,
            "compiler_verification": compiler_verification
        })

        pattern = DefaultPattern(
            initial_agent=outliner,
            agents=[paper_writer, outliner],
            context_variables=initial_context,
            group_manager_args={"llm_config": llm_config}
        )

        # ========================================================================
        # Step 1: Parse Outline and Dispatch to Abstract Generation
        # ========================================================================
        def parse_outline_and_dispatch(output, ctx: ContextVariables) -> FunctionTargetResult:
            """Parse outline JSON and move to abstract generation"""
            content = str(output)

            try:
                outline_data = parse_json_from_response(content)
                ctx["outline_structure"] = outline_data
                
                log_stage(workspace_dir, "report_generation",
                         f"Outline generated with {len(outline_data.get('sections', []))} sections")

                ctx["writing_mode"] = "abstract"
                paper_writer.handoffs.set_after_work(FunctionTarget(parse_abstract_and_dispatch))
                
                # Build outline sections summary for abstract prompt
                sections = outline_data.get('sections', [])
                outline_sections_str = "\n".join([
                    f"- {s['title']}: {s['description']}" 
                    for s in sections
                ])
                
                abstract_prompt_text = abstract_initial_prompt(
                    idea=ctx.get("idea", ""),
                    method_summary=ctx.get("method_summary", ""),
                    results_summary=ctx.get("results_summary", ""),
                    outline_sections=outline_sections_str
                )
                
                return FunctionTargetResult(
                    target=AgentTarget(paper_writer),
                    messages=abstract_prompt_text,
                    context_variables=ctx
                )

            except Exception as e:
                raise WorkflowError(f"Failed to parse outline: {str(e)}")

        # ========================================================================
        # Step 2: Parse Abstract and Dispatch to Self-Reflection
        # ========================================================================
        def parse_abstract_and_dispatch(output, ctx: ContextVariables) -> FunctionTargetResult:
            """Parse abstract JSON, perform self-reflection, and move to keywords generation"""
            content = str(output)

            try:
                abstract_data = parse_json_from_response(content)
                title = abstract_data.get("title", "Research Paper")
                abstract_text = abstract_data.get("abstract", "")
                
                ctx["paper_metadata"]["title"] = title
                ctx["paper_metadata"]["abstract"] = abstract_text

                # # Perform self-reflection to improve abstract
                # reflection_prompt = abstract_prompt(
                #     title=title,
                #     abstract=abstract_text,
                #     idea=ctx.get("idea", ""),
                #     method_summary=ctx.get("method_summary", ""),
                #     results_summary=ctx.get("results_summary", ""),
                #     output_type=output_type
                # )
                
                # # Update handoff to process reflection and then move to keywords
                # paper_writer.handoffs.set_after_work(FunctionTarget(process_abstract_reflection))
                
                # return FunctionTargetResult(
                #     target=AgentTarget(paper_writer),
                #     messages=reflection_prompt,
                #     context_variables=ctx
                # )
                    
                # Switch to keywords generation (keywords are extracted from abstract)
                ctx["writing_mode"] = "keywords"
                ctx["abstract_parse_attempts"] = 0
                paper_writer.handoffs.set_after_work(FunctionTarget(parse_keywords_and_dispatch))
                
                # Generate keywords prompt based on abstract and title
                keywords_prompt_text = keywords_prompt(
                    abstract=abstract_text,
                    title=ctx["paper_metadata"].get("title", "")
                )
                
                return FunctionTargetResult(
                    target=AgentTarget(paper_writer),
                    messages=keywords_prompt_text,
                    context_variables=ctx
                )

            except Exception as e:
                raise WorkflowError(f"Failed to parse abstract: {str(e)}")

        # ========================================================================
        # Step 3: Process Abstract Reflection and Dispatch to Keywords Generation
        # ========================================================================
        def process_abstract_reflection(output, ctx: ContextVariables) -> FunctionTargetResult:
            """Process abstract reflection and move to keywords generation"""
            content = str(output)
            
            # Extract improved abstract
            improved_abstract = _extract_latex_block(content, "Abstract")
            if improved_abstract == content.strip():
                improved_abstract = _extract_latex_block(content, "Text")
            
            # Fix percent symbols
            improved_abstract = _fix_percent(improved_abstract)
            
            # Update abstract in metadata
            ctx["paper_metadata"]["abstract"] = improved_abstract
            
            log_stage(workspace_dir, "report_generation",
                     f"Abstract improved through self-reflection")

            # Switch to keywords generation (keywords are extracted from abstract)
            ctx["writing_mode"] = "keywords"
            ctx["abstract_parse_attempts"] = 0
            paper_writer.handoffs.set_after_work(FunctionTarget(parse_keywords_and_dispatch))
            
            # Generate keywords prompt based on abstract and title
            keywords_prompt_text = keywords_prompt(
                abstract=improved_abstract,
                title=ctx["paper_metadata"].get("title", "")
            )
            
            return FunctionTargetResult(
                target=AgentTarget(paper_writer),
                messages=keywords_prompt_text,
                context_variables=ctx
            )

        # ========================================================================
        # Step 4: Parse Keywords and Dispatch to Section Writing
        # ========================================================================
        def parse_keywords_and_dispatch(output, ctx: ContextVariables) -> FunctionTargetResult:
            """Parse keywords JSON and move to section writing"""
            content = str(output)

            try:
                keywords_data = parse_json_from_response(content)
                keywords = keywords_data.get("keywords", "")
                ctx["paper_metadata"]["keywords"] = keywords
                
                log_stage(workspace_dir, "report_generation",
                         f"Keywords generated: {keywords}")

                # Switch to section writing mode
                ctx["writing_mode"] = "section"
                paper_writer.handoffs.set_after_work(FunctionTarget(section_completion_handler))
                
                return section_dispatcher(output, ctx)

            except Exception as e:
                raise WorkflowError(f"Failed to parse keywords: {str(e)}")

        # ========================================================================
        # Step 5: Section Dispatcher (Dispatch to Next Section)
        # ========================================================================

        def section_dispatcher(output, ctx: ContextVariables) -> FunctionTargetResult:
            """Dispatch to next section or terminate if all complete"""
            current_idx = ctx.get("current_section_index", 0)
            outline = ctx.get("outline_structure", {})
            sections = outline.get("sections", [])

            #history_messages

            if current_idx >= len(sections):
                # All sections complete, terminate
                return FunctionTargetResult(
                    target=TerminateTarget(),
                    context_variables=ctx
                )

            # Get current section
            section = sections[current_idx]
            paper_meta = ctx.get("paper_metadata", {})
            completed = ctx.get("completed_sections", {})

            # Use intelligent summarization: keep full content for recent sections, summaries for older ones
            prev_sections_summary = []
            prev_sections_content = {}
            keep_full_count = 100  # Keep full content for last 2 sections
            start_idx = max(0, current_idx - keep_full_count - 1)  # Include one more for summary
            
            for i in range(start_idx, current_idx):
                if i in completed:
                    section_title = sections[i]['title']
                    if i >= current_idx - keep_full_count:
                        # Keep full content for recent sections
                        prev_sections_content[section_title] = completed[i].get('content', '')
                        prev_sections_summary.append(
                            f"Section {i+1} ({section_title}): {prev_sections_content[section_title]}"
                        )
                    else:
                        # Use summary for older sections
                        summary = completed[i].get('summary', completed[i].get('content', '')[:200])
                        prev_sections_summary.append(
                            f"Section {i+1} ({section_title}): {summary}"
                        )
            
            prev_sections_str = "\n".join(prev_sections_summary) if prev_sections_summary else "No previous sections"


            # Use generic section prompt for other sections
            prompt = section_prompt(
                section_id=section["id"],
                section_title=section["title"],
                section_description=section["description"],
                section_guidelines=section["guidelines"],
                title=paper_meta.get("title", ""),
                abstract=paper_meta.get("abstract", ""),
                completed_sections=prev_sections_str,
                task=ctx.get("task", ""),
                idea=ctx.get("idea", ""),
                method_summary=ctx.get("method_summary", ""),
                results_summary=ctx.get("results_summary", ""),
                output_type=output_type
            )

            log_stage(workspace_dir, "report_generation",
                     f"Writing section {current_idx + 1}/{len(sections)}: {section['title']}")

            return FunctionTargetResult(
                target=AgentTarget(paper_writer),
                messages=prompt,
                context_variables=ctx
            )
        
        # Step 5.5 Hook Function: Manage the histroy messages of the paper writer
        def manage_history_messages(messages: list[dict]) -> list[dict]:
            """
            Only keep the system message and the last message for the paper writer.
            """
            ctx: ContextVariables = paper_writer.context_variables
            if ctx["writing_mode"] == "section":
                # The system message is in the default messages
                return [messages[-1]] # Last message
            else:
                return messages

        # ========================================================================
        # Step 6: Section Completion Handler (Save Section and Move to Next)
        # ========================================================================
        def section_completion_handler(output, ctx: ContextVariables) -> FunctionTargetResult:
            """Save section content and move to next"""
            current_idx = ctx.get("current_section_index", 0)
            outline = ctx.get("outline_structure", {})
            sections = outline.get("sections", [])
            section = sections[current_idx]

            content = str(output)
            # Extract LaTeX content from block if present
            section_title = section["title"]
            extracted_content = _extract_latex_block(content, section_title)
            if extracted_content != content.strip():
                content = extracted_content
            # Also try common section names as fallback
            if content == str(output).strip():
                for fallback_name in ["Section", "Text", section_title.replace(" ", "")]:
                    extracted = _extract_latex_block(str(output), fallback_name)
                    if extracted != str(output).strip():
                        content = extracted
                        break
            
            # Clean unwanted LaTeX wrappers
            content = _clean_section(content, section_title)
            
            # Fix special characters (especially %)
            content = _fix_percent(content)

            # Save section to file
            paper_dir = workspace_dir / "paper"
            sections_dir = paper_dir / "sections"
            sections_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{section['id']:02d}_{section['title'].lower().replace(' ', '_')}.tex"
            section_path = sections_dir / filename
            section_path.write_text(content, encoding='utf-8')

            # Compile and verify LaTeX, auto-fix if needed
            if ctx["compiler_verification"]:
                content = _compile_and_fix_section(
                    content, section_title, section_path, paper_dir, 
                    Journal.ICML2026, workspace_dir, paper_writer, llm_config
                )

            # Store summary and content in context
            completed = ctx.get("completed_sections", {})
            completed[current_idx] = {
                "title": section["title"],
                "summary": content[:200],  # Truncate to 200 chars
                "content": content,  # Store full content for next sections
                "path": str(section_path)
            }
            ctx["completed_sections"] = completed

            log_stage(workspace_dir, "report_generation",
                     f"[✓] Section {current_idx + 1} completed: {section['title']}")

            # Move to next section
            ctx["current_section_index"] = current_idx + 1
            return section_dispatcher(output, ctx)

        # Register the hook function
        paper_writer.register_hook(
                "process_all_messages_before_reply",
                manage_history_messages
            )
        # Set initial handoff for outline generation
        outliner.handoffs.set_after_work(FunctionTarget(parse_outline_and_dispatch))

        # Generate detailed outline prompt
        outline_prompt_text = outline_prompt(
            task=task,
            idea=selected_idea,
            method_summary=method_summary,
            results_summary=results_summary
        )

        paper_dir = workspace_dir / "paper"
        main_tex_path = paper_dir / "main.tex"
        # ========================================================================
        # Initiate group chat
        # ========================================================================
        if not skip_generation:
            log_stage(workspace_dir, "report_generation", "Starting report generation")
            log_stage(workspace_dir, "report_generation", "Generating paper outline")
            if state["config"]["researcher"]["iterable"]:
                result, context, last_agent = iterable_group_chat(
                    state,
                    max_rounds=len(initial_context.get("outline_structure", {}).get("sections", [])) + 5 if initial_context.get("outline_structure") else 20,
                    enable_hitl=False,
                    pattern=pattern,
                    prompt=outline_prompt_text,
                )
            else:
                result, context, last_agent = initiate_group_chat(
                    pattern=pattern,
                    messages=outline_prompt_text,
                    max_rounds=len(initial_context.get("outline_structure", {}).get("sections", [])) + 5 if initial_context.get("outline_structure") else 20
                )

            save_agent_history(
                workspace_dir=workspace_dir,
                node_name="report_generation",
                messages=result.chat_history,
                agent_chat_messages={
                    paper_writer.name: paper_writer.chat_messages
                }
            )

            # ========================================================================
            # Generate main.tex and references.bib files
            # ========================================================================
            
            paper_meta = context.get("paper_metadata", {})
            outline = context.get("outline_structure", {})

            keywords = paper_meta.get("keywords", "")
            main_tex_content = _generate_main_tex(
                paper_meta.get("title", "Research Paper"),
                paper_meta.get("abstract", ""),
                outline.get("sections", []),
                paper_dir,
                Journal.ICML2026,
                keywords=keywords
            )

            main_tex_path.write_text(main_tex_content, encoding='utf-8')

            references_bib_path = paper_dir / "references.bib"
            references_bib_path.write_text(bibliography_bib, encoding='utf-8')

            # Copy LaTeX template files to paper directory
            _copy_latex_template_files(paper_dir, Journal.ICML2026)

            log_stage(workspace_dir, "report_generation",
                    f"Completed. Generated {len(outline.get('sections', []))} sections")
            
            # Compile final PDF document
            log_stage(workspace_dir, "report_generation", "Compiling final document...")
        
        log_stage(workspace_dir, "report_generation", f"Skipping generation, saving report as format of {output_type}")

        try:
            if output_type == "pdf":
                bib_path = paper_dir / "references.bib"
                compile_success, error_msg = _run_latex_compilation(
                    main_tex_path, paper_dir, bib_path, timeout=60, cleanup=True
                )
                if compile_success:
                    log_stage(workspace_dir, "report_generation", 
                            "[✓] Final PDF document compiled successfully")
                else:
                    log_stage(workspace_dir, "report_generation", 
                            f"[!] Warning: Final PDF compilation had errors: {error_msg[:200] if error_msg else 'Unknown error'}")

            elif output_type == "markdown":
                from ..latex.latex_to_markdown import parse_main_tex
                paper_md_path = get_artifact_path(workspace_dir, "paper")
                content_md = parse_main_tex(main_tex_path)
                log_stage(workspace_dir, "report_generation", 
                         f"[✓] Markdown generated")

                save_markdown(content_md, paper_md_path)
                log_stage(workspace_dir, "report_generation", 
                         f"Markdown generated at: {paper_md_path}")
                paper_dir = paper_md_path
        except Exception as e:
            log_stage(workspace_dir, "report_generation", 
                     f"[!] Warning: Failed to compile final PDF: {str(e)}")

        # ========================================================================
        # Add citations if enabled
        # ========================================================================
        if include_references:
            api_key = citations_config.get("api_key", "")
            if api_key:
                log_stage(workspace_dir, "report_generation", "Adding citations to sections...")
                try:
                    max_paragraphs = citations_config.get("max_paragraphs_per_section", 50)
                    new_bib_entries = _add_citations_to_paper(
                        paper_dir,
                        outline.get("sections", []),
                        api_key,
                        max_paragraphs_per_section=max_paragraphs
                    )
                    
                    # Append new citations to existing bibliography
                    if new_bib_entries:
                        existing_bib = references_bib_path.read_text(encoding='utf-8')
                        combined_bib = existing_bib.rstrip() + '\n\n' + new_bib_entries
                        references_bib_path.write_text(combined_bib, encoding='utf-8')
                        log_stage(workspace_dir, "report_generation", 
                                 f"Added citations. New BibTeX entries added.")
                        
                        # Recompile PDF to include new citations
                        log_stage(workspace_dir, "report_generation", 
                                 "Recompiling PDF with new citations...")
                        try:
                            bib_path = paper_dir / "references.bib"
                            compile_success, error_msg = _run_latex_compilation(
                                main_tex_path, paper_dir, bib_path, timeout=60, cleanup=True
                            )
                            if compile_success:
                                log_stage(workspace_dir, "report_generation", 
                                         "[✓] PDF recompiled successfully with new citations")
                            else:
                                log_stage(workspace_dir, "report_generation", 
                                         f"[!] Warning: PDF recompilation had errors: {error_msg[:200] if error_msg else 'Unknown error'}")
                        except Exception as e:
                            log_stage(workspace_dir, "report_generation", 
                                     f"[!] Warning: Failed to recompile PDF: {str(e)}")
                    else:
                        log_stage(workspace_dir, "report_generation", 
                                 "No new citations found.")
                except Exception as e:
                    log_stage(workspace_dir, "report_generation", 
                             f"Warning: Failed to add citations: {str(e)}")
            else:
                log_stage(workspace_dir, "report_generation", 
                         "Warning: include_references enabled but API key not provided.")
                

        update_state = {
            "paper_dir": str(paper_dir),
            "stage": "report_generation"
        }
        # router
        if state["config"]["researcher"]["workflow"] == "default":
            update_state["next_node"] = "review"

        return update_state

    except Exception as e:
        log_stage(workspace_dir, "report_generation", f"Error: {str(e)}")
        raise WorkflowError(f"Report generation failed: {str(e)}")

def _extract_latex_block(text: str, block_name: str) -> str:
    """Extract LaTeX content from text between \\begin{block_name} and \\end{block_name}"""
    if isinstance(text, list):
        text = "".join([str(item) for item in text])
    
    pattern = rf"\\begin{{{block_name}}}(.*?)\\end{{{block_name}}}"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        # Fallback: return original text if block markers not found
        return str(text).strip()


def _clean_section(text: str, section_name: str) -> str:
    """Clean unwanted LaTeX wrappers from section content"""
    text = text.replace(r"\documentclass{article}", "")
    text = text.replace(r"\begin{document}", "")
    text = text.replace(r"\end{document}", "")
    text = text.replace(fr"\section{{{section_name}}}", "")
    text = text.replace(fr"\section*{{{section_name}}}", "")
    text = text.replace(fr"\begin{{{section_name}}}", "")
    text = text.replace(fr"\end{{{section_name}}}", "")
    text = text.replace(r"\maketitle", "")
    text = text.replace(r"```latex", "")
    text = text.replace(r"```", "")
    text = text.replace(r"\usepackage{amsmath}", "")
    # md
    text = text.replace(r"```markdown", "")
    return text.strip()


def _fix_percent(text: str) -> str:
    """Replace % (that is not \%) by \% in LaTeX text"""
    return re.sub(r'(?<!\\)%', r'\\%', text)


def _extract_selected_idea(idea_content: str) -> str:
    """Extract selected idea from idea.md"""
    selected_pattern = r'## Idea \d+ \*\*\[SELECTED\]\*\*\nScore: [\d.]+\nRound: \d+\n\n(.*?)(?=\n\n##|\n\n###|\Z)'
    match = re.search(selected_pattern, idea_content, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        # Fallback: return first idea
        first_idea_pattern = r'## Idea 1.*?\n\n(.*?)(?=\n\n##|\n\n###|\Z)'
        match = re.search(first_idea_pattern, idea_content, re.DOTALL)
        return match.group(1).strip() if match else idea_content[:500]


def _parse_literature_to_bibtex(workspace_dir: Path) -> tuple[str, List[Dict[str, str]]]:
    """Parse literature data from arxiv_cache/*_metadata.json files to BibTeX format and extract cite keys"""
    bib_entries = []
    cite_keys_info = []

    arxiv_cache_dir = workspace_dir / "literature" / "arxiv_cache"
    if not arxiv_cache_dir.exists():
        return "", []
    
    metadata_files = sorted(arxiv_cache_dir.glob("*_metadata.json"))
    
    if not metadata_files:
        return "", []
    
    seen_arxiv_ids = set()  # Avoid duplicates across multiple metadata files
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            papers = cache_data.get("papers", [])
            for paper in papers:
                arxiv_id = paper.get("arxiv_id", "")
                if not arxiv_id or arxiv_id in seen_arxiv_ids:
                    continue
                seen_arxiv_ids.add(arxiv_id)
                
                title = paper.get("title", "")
                authors_list = paper.get("authors", [])
                year = paper.get("year")
                url = paper.get("url", "")
                
                if not title:
                    continue
                
                # Generate cite key: firstauthor_year_firstword
                if authors_list and len(authors_list) > 0:
                    first_author = authors_list[0].split(',')[0].strip() if isinstance(authors_list[0], str) else str(authors_list[0]).split(',')[0].strip()
                else:
                    first_author = "unknown"
                
                first_word = re.sub(r'[^a-zA-Z]', '', title.split()[0].lower()) if title.split() else "paper"
                year_str = str(year) if year else "unknown"
                cite_key = f"{first_author.lower().replace(' ', '')}{year_str}{first_word}"
                
                # Format authors for BibTeX
                if isinstance(authors_list, list):
                    authors_bib = " and ".join([str(a).strip() for a in authors_list])
                else:
                    authors_bib = "Unknown"
                
                # Generate BibTeX entry
                bib_entry = (
                    f"@article{{{cite_key},\n"
                    f"  title={{{title}}},\n"
                    f"  author={{{authors_bib}}},\n"
                )
                if year:
                    bib_entry += f"  year={{{year}}},\n"
                if url:
                    bib_entry += f"  url={{{url}}},\n"
                bib_entry += f"  note={{arXiv:{arxiv_id}}}\n"
                bib_entry += "}"
                bib_entries.append(bib_entry)

                # Store cite key info for prompt
                cite_keys_info.append({
                    "key": cite_key,
                    "title": title,
                    "authors": ", ".join(authors_list) if isinstance(authors_list, list) else "Unknown",
                    "year": str(year) if year else "Unknown"
                })
        except Exception as e:
            # Skip invalid files but continue processing others
            continue

    bib_content = "\n\n".join(bib_entries) if bib_entries else ""
    return bib_content, cite_keys_info


def _generate_main_tex(paper_title: str, paper_abstract: str, sections: List[Dict], 
                       paper_dir: Path, journal: Journal = Journal.ICML2026, 
                       keywords: str = "") -> str:
    """Generate main.tex file content using journal template"""
    preset = get_preset(journal)
    
    section_includes = []
    for section in sections:
        section_id = section["id"]
        section_title = section["title"]
        filename = f"{section_id:02d}_{section_title.lower().replace(' ', '_')}.tex"
        section_includes.append(f"\\input{{sections/{filename}}}")

    # Default author and affiliation
    author = "Research Team"
    affiliation = "Research Institution"

    # Build LaTeX document using preset
    layout_str = f"[{preset.layout}]" if preset.layout else ""
    usepackage_str = f"\n{preset.usepackage}\n" if preset.usepackage else ""
    
    # Escape LaTeX special characters in title and abstract
    paper_title_escaped = paper_title.replace("&", "\\&").replace("%", "\\%")
    paper_abstract_escaped = paper_abstract.replace("&", "\\&").replace("%", "\\%")
    
    # Add keywords if provided
    keywords_str = f"\n{preset.keywords(keywords)}\n" if keywords and preset.keywords else ""
    
    main_tex = (
        f"\\documentclass{layout_str}{{{preset.article}}}\n\n"
        f"% Packages\n"
        f"\\usepackage{{amsmath,amssymb}}\n"
        f"\\usepackage{{graphicx}}\n"
        f"\\usepackage{{hyperref}}\n"
        f"\\usepackage{{natbib}}\n"
        f"{usepackage_str}"
        f"\n\\begin{{document}}\n\n"
        f"{preset.title}{{{paper_title_escaped}}}\n"
        f"{preset.author(author)}\n"
        f"{preset.affiliation(affiliation)}\n"
        f"{preset.abstract(paper_abstract_escaped)}\n"
        f"{keywords_str}"
        f"\n% Sections\n"
        f"{chr(10).join(section_includes)}\n\n"
        f"% Bibliography\n"
        f"\\bibliography{{references}}\n"
        f"{preset.bibliographystyle}\n\n"
        f"\\end{{document}}\n"
    )
    return main_tex


def _copy_latex_template_files(paper_dir: Path, journal: Journal) -> None:
    """Copy LaTeX template files to paper directory"""
    preset = get_preset(journal)
    # Path to latex template directory: researcher/latex/icml2026/
    latex_source_dir = Path(__file__).parent.parent / "latex" / journal.value.lower()
    
    if not latex_source_dir.exists():
        log_stage(paper_dir.parent, "report_generation", 
                 f"Warning: LaTeX template directory not found: {latex_source_dir}")
        return
    
    # Copy all required template files
    for filename in preset.files:
        source_file = latex_source_dir / filename
        if source_file.exists():
            dest_file = paper_dir / filename
            shutil.copy2(source_file, dest_file)
            log_stage(paper_dir.parent, "report_generation",
                     f"Copied template file: {filename}")
        else:
            log_stage(paper_dir.parent, "report_generation",
                     f"Warning: Template file not found: {source_file}")


def _run_latex_compilation(tex_file: Path, work_dir: Path, bib_path: Optional[Path] = None,
                           timeout: int = 60, cleanup: bool = True, 
                           cleanup_extensions: Optional[List[str]] = None) -> tuple[bool, Optional[str]]:
    """
    Generic LaTeX compilation function using xelatex and bibtex.
    
    Args:
        tex_file: Path to the .tex file to compile
        work_dir: Working directory for compilation
        bib_path: Optional path to bibliography file
        timeout: Timeout for each compilation step
        cleanup: Whether to clean up auxiliary files after compilation
        cleanup_extensions: List of file extensions to clean (default: common aux files)
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    tex_name = tex_file.name
    tex_stem = tex_file.stem
    has_bibliography = bib_path is not None and bib_path.exists()
    
    def run_xelatex():
        return subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-file-line-error", tex_name],
            cwd=work_dir,
            input="\n",
            capture_output=True,
            text=True,
            timeout=timeout
        )
    
    def run_bibtex():
        return subprocess.run(
            ["bibtex", tex_stem],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
    
    try:
        # Pass 1: First xelatex run
        result = run_xelatex()
        if result.returncode != 0:
            error_msg = _extract_latex_errors(result.stdout, result.stderr)
            return False, error_msg
        
        # Pass 2: Run bibtex
        if has_bibliography:
            bib_result = run_bibtex()
            if bib_result.returncode != 0:
                # BibTeX errors are usually non-fatal, continue compilation
                pass
            
            # Pass 3 & 4: Additional xelatex runs for citations and cross-references
            for _ in range(2):
                result = run_xelatex()
                if result.returncode != 0:
                    error_msg = _extract_latex_errors(result.stdout, result.stderr)
                    return False, error_msg
        else:
            # Pass 2: Second xelatex run for cross-references even without bibliography
            result = run_xelatex()
            if result.returncode != 0:
                error_msg = _extract_latex_errors(result.stdout, result.stderr)
                return False, error_msg
        
        if cleanup:
            if cleanup_extensions is None:
                cleanup_extensions = ['.aux', '.log', '.out', '.bbl', '.blg', '.synctex.gz']
            for ext in cleanup_extensions:
                aux_file = tex_file.with_suffix(ext)
                if aux_file.exists():
                    aux_file.unlink()
        
        return True, None
        
    except subprocess.TimeoutExpired:
        return False, "LaTeX compilation timeout"
    except Exception as e:
        return False, f"Compilation error: {str(e)}"


def _create_test_document(section_content: str, section_name: str, paper_dir: Path, 
                          journal: Journal) -> Path:
    """Create a temporary test LaTeX document for section validation"""
    preset = get_preset(journal)
    temp_dir = paper_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a minimal test document
    layout_str = f"[{preset.layout}]" if preset.layout else ""
    usepackage_str = f"\n{preset.usepackage}\n" if preset.usepackage else ""
    
    test_doc = (
        f"\\documentclass{layout_str}{{{preset.article}}}\n\n"
        f"\\usepackage{{amsmath,amssymb}}\n"
        f"\\usepackage{{graphicx}}\n"
        f"\\usepackage{{hyperref}}\n"
        f"\\usepackage{{natbib}}\n"
        f"{usepackage_str}"
        f"\n\\begin{{document}}\n\n"
        f"\\section{{{section_name}}}\n"
        f"{section_content}\n"
        f"\n\\end{{document}}\n"
    )
    
    test_file = temp_dir / f"test_{section_name.lower().replace(' ', '_')}.tex"
    test_file.write_text(test_doc, encoding='utf-8')
    
    # Copy required template files to temp directory
    latex_source_dir = Path(__file__).parent.parent / "latex" / journal.value.lower()
    for filename in preset.files:
        source_file = latex_source_dir / filename
        if source_file.exists():
            shutil.copy2(source_file, temp_dir / filename)
    
    return test_file


def _extract_latex_errors(stdout: str, stderr: str) -> str:
    """Extract LaTeX compilation errors from output"""
    error_lines = []
    lines = (stdout + "\n" + stderr).split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("! "):
            error_block = [line]
            i += 1
            # Capture following error context
            while i < len(lines) and i < 10:  # Limit context
                next_line = lines[i].strip()
                if (next_line.startswith("! ") or 
                    next_line.startswith("l.") or
                    next_line.startswith("(") or
                    next_line == ""):
                    if next_line.startswith("l."):
                        error_block.append(next_line)
                    break
                error_block.append(next_line)
                i += 1
            error_lines.append("\n".join(error_block))
        i += 1
    
    return "\n\n".join(error_lines) if error_lines else "Unknown LaTeX error"


def _fix_latex_with_llm(section_content: str, section_name: str, error_message: str,
                        paper_writer, llm_config, workspace_dir: Path) -> str:
    """Use LLM to fix LaTeX compilation errors"""
    from autogen.agentchat import initiate_group_chat
    from autogen.agentchat.group.patterns import DefaultPattern
    from autogen.agentchat.group import ContextVariables, AgentTarget, FunctionTargetResult
    
    prompt = fix_latex_prompt(section_content, error_message, section_name)
    
    pattern = DefaultPattern(
        initial_agent=paper_writer,
        agents=[paper_writer],
        context_variables=ContextVariables(data={}),
        group_manager_args={"llm_config": llm_config}
    )
    
    result, context, last_agent = initiate_group_chat(
        pattern=pattern,
        messages=prompt,
        max_rounds=3
    )
    
    # Extract fixed content
    fixed_content = _extract_latex_block(str(result), "Text")
    if fixed_content == str(result).strip():
        # Fallback: try "Section" or section name
        for block_name in ["Section", section_name, "Text"]:
            fixed_content = _extract_latex_block(str(result), block_name)
            if fixed_content != str(result).strip():
                break
    
    return fixed_content if fixed_content != str(result).strip() else section_content


def _compile_and_fix_section(section_content: str, section_name: str, section_path: Path,
                             paper_dir: Path, journal: Journal, workspace_dir: Path,
                             paper_writer, llm_config: Dict[str, Any],
                             max_attempts: int = 3) -> str:
    """Compile LaTeX section and auto-fix errors if needed"""
    temp_dir = paper_dir / "temp"
    
    def compile_section(content: str) -> tuple[bool, Optional[str]]:
        """Helper to compile section content"""
        test_file = _create_test_document(content, section_name, paper_dir, journal)
        bib_path = temp_dir / "references.bib"
        cleanup_exts = ['.aux', '.log', '.out', '.pdf', '.bbl', '.blg', '.synctex.gz']
        return _run_latex_compilation(test_file, temp_dir, bib_path, timeout=30, 
                                      cleanup=True, cleanup_extensions=cleanup_exts)
    
    # First attempt: compile
    success, error_msg = compile_section(section_content)
    
    if success:
        log_stage(workspace_dir, "report_generation", 
                 f"[✓] Section '{section_name}' compiled successfully")
        return section_content
    
    log_stage(workspace_dir, "report_generation", 
             f"[!] Section '{section_name}' has LaTeX errors, attempting to fix...")
    
    # Try basic fixes first: escape % if not already escaped
    fixed_content = _fix_percent(section_content)
    success, error_msg = compile_section(fixed_content)
    
    if success:
        log_stage(workspace_dir, "report_generation", 
                 f"[✓] Section '{section_name}' fixed with basic corrections")
        section_path.write_text(fixed_content, encoding='utf-8')
        return fixed_content
    
    # If basic fixes didn't work, try LLM-based fixing
    for attempt in range(max_attempts):
        log_stage(workspace_dir, "report_generation", 
                 f"  Attempting LLM fix (attempt {attempt + 1}/{max_attempts})...")
        
        try:
            fixed_content = _fix_latex_with_llm(
                fixed_content, section_name, error_msg, 
                paper_writer, llm_config, workspace_dir
            )
            
            # Clean and fix percent again
            fixed_content = _clean_section(fixed_content, section_name)
            fixed_content = _fix_percent(fixed_content)
            
            # Try compiling again
            success, new_error_msg = compile_section(fixed_content)
            
            if success:
                log_stage(workspace_dir, "report_generation", 
                         f"[✓] Section '{section_name}' fixed with LLM assistance")
                section_path.write_text(fixed_content, encoding='utf-8')
                return fixed_content
            else:
                error_msg = new_error_msg  # Update error for next attempt
                
        except Exception as e:
            log_stage(workspace_dir, "report_generation", 
                     f"  LLM fix attempt {attempt + 1} failed: {str(e)}")
    
    # If all attempts failed, return the best we have
    log_stage(workspace_dir, "report_generation", 
             f"[!] Could not fully fix '{section_name}' after {max_attempts} attempts, keeping best version")
    section_path.write_text(fixed_content, encoding='utf-8')
    return fixed_content


# Citation Addition Functions
def _execute_perplexity_query(payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Execute a query to Perplexity API for citation search."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def _add_references_with_perplexity(paragraph: str, api_key: str) -> Tuple[str, List[str]]:
    """Add references to a paragraph using Perplexity API."""
    perplexity_message = citation_addition_prompt(paragraph)
    
    payload = {
        "model": "sonar-reasoning-pro",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Be precise and concise. Follow the instructions."},
            {"role": "user", "content": perplexity_message}
        ],
        "search_domain_filter": ["arxiv.org"],
    }
    
    try:
        perplexity_response = _execute_perplexity_query(payload, api_key)
        content = perplexity_response["choices"][0]["message"]["content"]
        citations = perplexity_response.get("citations", [])
        cleaned_response = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
        return cleaned_response, citations
    except Exception:
        return paragraph, []


def _extract_paragraphs_from_tex_for_citations(tex_content: str) -> Dict[int, str]:
    """Extract paragraph-like lines from LaTeX content for citation processing."""
    paragraph_lines = {}
    lines = tex_content.splitlines(keepends=True)

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith('%'):
            continue

        if re.match(r'\\(begin|end|section|subsection|subsubsection|label|caption|ref|title|author|documentclass|usepackage|newcommand|section|subsection|affiliation|keywords|bibliography|centering|includegraphics)', line):
            continue

        if re.search(r'\\(item|enumerate)', line):
            continue

        if re.search(r'(figure|table|equation|align|tabular)', line):
            continue

        if re.match(r'^\$.*\$$', line) or re.match(r'^\\\[.*\\\]$', line):
            continue

        paragraph_lines[i] = raw_line

    return paragraph_lines


def _arxiv_url_to_bibtex(citations: List[str]) -> Tuple[List[str], List[str]]:
    """
    Convert arXiv URLs to BibTeX entries.

    Args:
        citations (List[str]): List of arXiv URLs (abs, pdf, or html variants allowed).

    Returns:
        Tuple[List[str], List[str]]:
            - A list of BibTeX keys (as strings).
            - A list of full BibTeX entries (as strings) suitable for inclusion in a .bib file.
    """
    bib_keys = []
    bib_strs = []

    for url in citations:
        try:
            # Convert URL to bibtex url (e.g., from /abs/ or /html/ to /bibtex/)
            bib_url = re.sub(r'\b(abs|html|pdf)\b', 'bibtex', url)
            response = requests.get(bib_url, timeout=10)

            # If fetching fails, try the fallback using the arXiv ID
            if response.status_code != 200:
                # Extract arXiv id from the URL (matches patterns like 2010.07487)
                match_id = re.search(r'(\d{4}\.\d+)', url)
                if match_id:
                    arxiv_id = match_id.group(1)
                    fallback_url = f"https://arxiv.org/bibtex/{arxiv_id}"
                    response = requests.get(fallback_url, timeout=10)
                    if response.status_code != 200:
                        # Fallback failed; mark this citation as failed.
                        bib_keys.append(None)
                        continue
                else:
                    # Could not extract arXiv id; mark as failed.
                    bib_keys.append(None)
                    continue

            bib_str = response.text.strip()

            # Extract BibTeX key using regex
            match = re.match(r'@[\w]+\{([^,]+),', bib_str)
            if not match:
                # Could not extract key; mark as failed.
                bib_keys.append(None)
                continue

            bib_key = match.group(1)
            bib_keys.append(bib_key)
            bib_strs.append(bib_str)

        except Exception:
            bib_keys.append(None)
            continue

    return bib_keys, bib_strs


def _replace_citations_with_latex(content: str, bib_keys: List[str]) -> str:
    """Replace [1][2][3] style citations with LaTeX \citep{} format."""
    def extract_year(key: str) -> int:
        match = re.search(r'\d{4}', key)
        return int(match.group()) if match else float('inf')
    
    def replacer(match):
        numbers = re.findall(r'\[(\d+)\]', match.group())
        keys = [bib_keys[int(n) - 1] for n in numbers if int(n) - 1 < len(bib_keys) and bib_keys[int(n) - 1] is not None]
        if not keys:
            return ""
        sorted_keys = sorted(keys, key=extract_year)
        return f" \\citep{{{','.join(sorted_keys)}}}"
    
    pattern = r'(?:\[\d+\])+'
    return re.sub(pattern, replacer, content)


def _process_section_with_citations(
    section_text: str,
    api_key: str,
    max_paragraphs: Optional[int] = None
) -> Tuple[str, str]:
    """Process a LaTeX section by adding citations."""
    lines = section_text.splitlines()
    para_dict = _extract_paragraphs_from_tex_for_citations(section_text)
    
    str_bib = ''
    count = 0
    
    for kpara in sorted(para_dict.keys()):
        if count == 0:
            count += 1
            continue
        
        if max_paragraphs is not None and count >= max_paragraphs:
            break
        
        para = para_dict[kpara]
        
        for attempt in range(2):
            try:
                new_para, citations = _add_references_with_perplexity(para, api_key)
                if new_para and citations:
                    break
            except Exception:
                if attempt == 1:
                    new_para = para
                    citations = []
                continue
        
        if citations:
            bib_keys, bib_strs = _arxiv_url_to_bibtex(citations)
            new_para = _replace_citations_with_latex(new_para, bib_keys)
            
            for bib_str in bib_strs:
                if bib_str and bib_str not in str_bib:
                    str_bib = str_bib.rstrip() + '\n\n' + bib_str if str_bib else bib_str
        
        lines[kpara] = new_para
        count += 1
    
    new_text = ''.join(lines)
    return new_text, str_bib


def _add_citations_to_paper(
    paper_dir: Path,
    sections: List[Dict[str, Any]],
    api_key: str,
    max_paragraphs_per_section: Optional[int] = None
) -> str:
    """Add citations to all sections of a paper"""
    sections_dir = paper_dir / "sections"
    all_bib_entries = []
    bib_entries_set = set()
    
    for section in sections:
        section_id = section["id"]
        section_title = section["title"]
        filename = f"{section_id:02d}_{section_title.lower().replace(' ', '_')}.tex"
        section_path = sections_dir / filename
        
        if not section_path.exists():
            continue
        
        try:
            section_text = section_path.read_text(encoding='utf-8')
            updated_text, bib_entries = _process_section_with_citations(
                section_text,
                api_key,
                max_paragraphs=max_paragraphs_per_section
            )
            
            section_path.write_text(updated_text, encoding='utf-8')
            
            if bib_entries:
                entries = bib_entries.strip().split('\n\n')
                for entry in entries:
                    clean_entry = entry.strip()
                    if clean_entry and clean_entry not in bib_entries_set:
                        bib_entries_set.add(clean_entry)
                        all_bib_entries.append(clean_entry)
        except Exception:
            continue
    
    return "\n\n".join(all_bib_entries)

