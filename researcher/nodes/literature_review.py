from typing import Dict, Any, List, Annotated, Optional
from pathlib import Path

from autogen import ConversableAgent, ContextExpression
from autogen.agentchat import initiate_group_chat, register_function
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group import (
    AgentTarget,
    FunctionTarget,
    FunctionTargetResult,
    TerminateTarget,
    ContextVariables,
    ReplyResult,
    OnContextCondition,
    ExpressionContextCondition
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

        def search_arxiv_papers(
            context_variables: ContextVariables,
            query: Annotated[str, "Search query for arxiv academic database"],
            max_results: Annotated[int, "Maximum number of papers to retrieve"] = 5,
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
                        "authors": [author.name for author in result.authors],
                        "abstract": result.summary,
                        "year": result.published.year if result.published else None,
                        "query": query,
                        "pdf_file": pdf_filename,
                        "url": result.entry_id
                    })

                if cache_metadata["papers"]:
                    metadata_path = cache_dir / f"{timestamp}_metadata.json"
                    save_json(cache_metadata, metadata_path)

                # Format papers text
                papers_text = "\n\n---\n\n".join(papers_text_parts) if papers_text_parts else "No papers found"

                context_variables["state"] = "searched"
                context_variables["searching_results"].append({
                    "success": True,
                    "papers": papers,
                    "query": query,
                    "formatted_text": papers_text
                })


                # return {
                #     "success": True,
                #     "papers": papers,
                #     "query": query,
                #     "formatted_text": papers_text
                # }
                return ReplyResult(
                    message="Searching complete.",
                    #target=FunctionTarget(format_papers_for_summarizer),
                    context_variables=context_variables
                )

            except ImportError:
                return {"success": False, "error": "arxiv package not installed"}
            except Exception as e:
                return {"success": False, "error": str(e)}



        # 1. 定义 hook 函数
        def inject_context_to_prompt(messages: list[dict]) -> list[dict]:
            """
            把 context variables 里的 task 和 papers 信息，注入到 Summarizer 的 prompt 里。
            """
            # messages - 当前 agent 的对话历史（含 user + system 等）
            # 先查最后一条 user 说的话
            last_user = None
            if messages:
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        last_user = msg
                        break

            # 读取 context variables
            # 当前 registered hook 是挂在 Summarizer agent 上，
            # 所以我们可以通过 messages 找 owner agent 的 context
            # 假设 Summarizer agent 有 context_variables 属性
            # 你也可以通过闭包传进来
            ctx: ContextVariables = summarizer.context_variables

            task_text = ctx.get("task", "")
            searching_results = ctx.get("searching_results", [])

            # 拼接所有检索到的论文摘要
            papers_text_list = []
            for r in searching_results:
                formatted = r.get("formatted_text", "")
                if formatted:
                    papers_text_list.append(formatted)

            combined_papers = "\n\n---\n\n".join(papers_text_list) if papers_text_list else ""

            # 构造额外提示
            extra_prompt = (
                f"Task:\n{task_text}\n\n"
                f"Papers:\n{combined_papers}\n\n"
                "Use the above task and paper info to generate a summarized synthesis."
            )

            # 我们把这个 prompt 放在 messages 最后 user 之前
            # 这样 LLM 在看到 Summarizer 输入时会包含我们动态构造的内容
            new_msgs = []
            for msg in messages:
                new_msgs.append(msg)

            # 插入作为额外 user prompt
            new_msgs.append({
                "role": "user",
                "content": extra_prompt
            })

            return new_msgs



        

        searcher = LiteratureSearcherAgent().create_agent(llm_config)
        summarizer = LiteratureSummarizerAgent().create_agent(llm_config)

        # def always_trigger(sender):
        #     return True
        #summarizer.register_reply(always_trigger, context_for_summarization_reply_func, position=0)
        
        llm_config_tool = get_llm_config(Path("configs/llm_config_tool.json"))
        executor = ConversableAgent(
            name="SearchExecutor",
            human_input_mode="NEVER",
            system_message="Initiate the tool call based on the received search keywords and other information",
            llm_config=llm_config_tool,
            functions=[search_arxiv_papers]
        )

        # register_function(
        #     search_arxiv_papers,
        #     caller=executor,
        #     executor=executor,
        #     description="Search arxiv for academic papers, download and cache PDFs, return paper metadata",
        # )

        initial_variables = ContextVariables(data={
            "state": "start",
            "task": task,
            "searching_results": []
        })

        pattern = DefaultPattern(
            initial_agent=searcher,
            agents=[searcher, executor, summarizer],
            group_manager_args={"llm_config": llm_config},
            context_variables=initial_variables
        )

        def format_papers_for_summarizer(output: any, ctx: ContextVariables):
            message = f"""{LITERATURE_SUMMARY_PROMPT.format(
                papers="Extract and synthesize the papers from the search results above (look for formatted_text in the tool results).",
                task=task
            )}"""
            
            return FunctionTargetResult(
                target=AgentTarget(summarizer),
                messages=message,
                context_variables=ctx
            )

        searcher.handoffs.set_after_work(AgentTarget(executor))
        executor.handoffs.add_context_condition(
            OnContextCondition(
                #target=FunctionTarget(format_papers_for_summarizer),
                target=AgentTarget(summarizer),
                condition=ExpressionContextCondition(ContextExpression("${state} == 'searched'"))
            )
        )
        summarizer.register_hook(
            "process_all_messages_before_reply",
            inject_context_to_prompt
        )
        summarizer.handoffs.set_after_work(TerminateTarget())

        prompt = LITERATURE_SEARCH_PROMPT.format(task=task)

        result, context, last_agent = initiate_group_chat(
            pattern=pattern,
            messages=prompt,
            max_rounds=10
        )

        # Extract papers and synthesis from messages
        literature_items = []
        synthesis = None
        
        # Get papers from context_variables["searching_results"]
        searching_results = context.get("searching_results", [])
        for result_data in searching_results:
            if result_data.get("success") and "papers" in result_data:
                for paper in result_data["papers"]:
                    literature_items.append(LiteratureItem(
                        title=paper.get("title", ""),
                        authors=paper.get("authors", []),
                        abstract=paper.get("abstract", ""),
                        url=paper.get("url", ""),
                        year=paper.get("year")
                    ))

        # Extract synthesis from messages (from summarizer)
        for msg in result.chat_history:
            content = msg.get("content", "")
            if msg.get("name") == summarizer.name and len(content) > 100:
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
        save_markdown(literature.to_markdown(), lit_path)

        log_stage(workspace_dir, "literature_review", f"Completed. Found {len(literature_items)} papers")

        return {"task": task, "literature": literature, "stage": "literature_review"}

    except Exception as e:
        log_stage(workspace_dir, "literature_review", f"Error: {str(e)}")
        raise WorkflowError(f"Literature review failed: {str(e)}")
