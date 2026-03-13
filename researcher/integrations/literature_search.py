from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import re

import requests

from researcher.utils import save_json
from researcher.exceptions import WorkflowError


def _extract_year(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\d{4}", value)
        if match:
            return int(match.group(0))
    return None


def _format_papers_text(papers: List[Dict[str, Any]]) -> str:
    if not papers:
        return "No papers found"
    parts = []
    for paper in papers:
        title = paper.get("title", "")
        authors = ", ".join(paper.get("authors", []))
        abstract = paper.get("abstract", "") or ""
        source = paper.get("source", "")
        url = paper.get("url", "")
        parts.append(
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Abstract: {abstract}\n"
            f"Source: {source}\n"
            f"URL: {url}\n"
        )
    return "\n\n---\n\n".join(parts)


def _search_arxiv_papers(query: str, max_results: int, workspace_dir: Path) -> List[Dict[str, Any]]:
    import arxiv

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    cache_dir = workspace_dir / "literature" / "arxiv_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for result in search.results():
        arxiv_id = result.entry_id.split('/')[-1]
        paper_data = {
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "url": result.entry_id,
            "year": result.published.year if result.published else None,
            "arxiv_id": arxiv_id,
            "source": "arxiv",
        }
        papers.append(paper_data)

        # Cache PDF
        pdf_filename = None
        try:
            pdf_filename = f"{timestamp}_{arxiv_id}.pdf"
            result.download_pdf(dirpath=str(cache_dir), filename=pdf_filename)
            paper_data["pdf_cached"] = pdf_filename
        except Exception:
            paper_data["pdf_cached"] = None

    return papers


def _search_pubmed_papers(query: str, max_results: int, api_key: str) -> List[Dict[str, Any]]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    esearch_url = f"{base}/esearch.fcgi"
    esummary_url = f"{base}/esummary.fcgi"

    esearch_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    if api_key:
        esearch_params["api_key"] = api_key
    esearch_resp = requests.get(esearch_url, params=esearch_params, timeout=20)
    esearch_resp.raise_for_status()
    esearch_data = esearch_resp.json()
    id_list = esearch_data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return []

    esummary_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "json",
    }
    if api_key:
        esummary_params["api_key"] = api_key
    esummary_resp = requests.get(esummary_url, params=esummary_params, timeout=20)
    esummary_resp.raise_for_status()
    esummary_data = esummary_resp.json().get("result", {})

    papers = []
    for pmid in esummary_data.get("uids", []):
        item = esummary_data.get(pmid, {})
        authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
        year = _extract_year(item.get("pubdate"))
        papers.append({
            "title": item.get("title", ""),
            "authors": authors,
            "abstract": "",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "year": year,
            "source": "pubmed",
            "external_id": pmid,
        })
    return papers


def _search_semantic_scholar_papers(query: str, max_results: int, api_key: str) -> List[Dict[str, Any]]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    params = {
        "query": query,
        "fields": "title,authors,year,abstract,url,openAccessPdf",
    }
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    resp = requests.get(url, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    papers = []
    for item in data[:max_results]:
        authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
        papers.append({
            "title": item.get("title", ""),
            "authors": authors,
            "abstract": item.get("abstract", "") or "",
            "url": item.get("url", "") or "",
            "year": item.get("year"),
            "source": "semantic_scholar",
            "external_id": item.get("paperId"),
            "pdf_url": (item.get("openAccessPdf") or {}).get("url"),
        })
    return papers


def _openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    if not inverted_index:
        return ""
    max_pos = 0
    for positions in inverted_index.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            if 0 <= pos < len(words):
                words[pos] = word
    return " ".join([w for w in words if w])


def _search_openalex_papers(query: str, max_results: int) -> List[Dict[str, Any]]:
    url = "https://api.openalex.org/works"
    params = {"search": query}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json().get("results", [])

    papers = []
    for item in data[:max_results]:
        authors = []
        for auth in item.get("authorships", []):
            name = auth.get("author", {}).get("display_name")
            if name:
                authors.append(name)
        abstract = _openalex_abstract(item.get("abstract_inverted_index"))
        url_value = item.get("id") or item.get("primary_location", {}).get("landing_page_url", "")
        papers.append({
            "title": item.get("display_name", ""),
            "authors": authors,
            "abstract": abstract,
            "url": url_value,
            "year": item.get("publication_year"),
            "source": "openalex",
            "external_id": item.get("id"),
        })
    return papers


def _search_crossref_papers(query: str, max_results: int) -> List[Dict[str, Any]]:
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": max_results}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    items = resp.json().get("message", {}).get("items", [])

    papers = []
    for item in items:
        authors = []
        for author in item.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")
            name = " ".join([part for part in [given, family] if part]).strip()
            if name:
                authors.append(name)
        year = None
        for key in ("published-print", "published-online", "issued"):
            date_parts = item.get(key, {}).get("date-parts", [])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
                break
        papers.append({
            "title": (item.get("title") or [""])[0],
            "authors": authors,
            "abstract": item.get("abstract", "") or "",
            "url": item.get("URL", ""),
            "year": year,
            "source": "crossref",
            "external_id": item.get("DOI"),
        })
    return papers



def _search_perplexity_papers(query: str, max_results: int, api_key: str) -> List[Dict[str, Any]]:
    if not api_key:
        raise WorkflowError("perplexity_api_key is required")

    url = "https://api.perplexity.ai/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "max_results": max_results,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=20)
    resp.raise_for_status()
    results = resp.json().get("results", [])

    papers = []
    for item in results:
        year = _extract_year(item.get("date"))
        papers.append({
            "title": item.get("title", ""),
            "authors": [],
            "abstract": item.get("snippet", "") or "",
            "url": item.get("url", ""),
            "year": year,
            "source": "perplexity",
            "external_id": item.get("url", ""),
        })
    return papers


def _cache_metadata(
    source: str,
    query: str,
    max_results: int,
    papers: List[Dict[str, Any]],
    workspace_dir: Path,
) -> None:
    if not papers:
        return
    cache_dir = workspace_dir / "literature" / f"{source}_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": timestamp,
        "query": query,
        "max_results": max_results,
        "source": source,
        "papers": papers,
    }
    metadata_path = cache_dir / f"{timestamp}_metadata.json"
    save_json(payload, metadata_path)



def search_literature(
    query: str,
    max_results: int,
    sources: List[str],
    workspace_dir: Path,
    api_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    semantic_scholar_api_key = api_config.get("semantic_scholar_api_key", "")
    pubmed_api_key = api_config.get("pubmed_api_key", "")
    perplexity_api_key = api_config.get("perplexity_api_key", "")

    normalized_sources = [source.lower() for source in sources]
    handlers = {
        "arxiv": lambda q, n: _search_arxiv_papers(q, n, workspace_dir),
        "pubmed": lambda q, n: _search_pubmed_papers(q, n, pubmed_api_key),
        "semantic_scholar": lambda q, n: _search_semantic_scholar_papers(q, n, semantic_scholar_api_key),
        "openalex": _search_openalex_papers,
        "crossref": _search_crossref_papers,
        "perplexity": lambda q, n: _search_perplexity_papers(q, n, perplexity_api_key),
    }

    results: List[Dict[str, Any]] = []
    for source in normalized_sources:
        handler = handlers.get(source)
        if not handler:
            raise WorkflowError(f"Unknown literature source: {source}")
        papers = handler(query, max_results)
        _cache_metadata(source, query, max_results, papers, workspace_dir)
        results.append({
            "success": True,
            "source": source,
            "papers": papers,
            "query": query,
            "formatted_text": _format_papers_text(papers),
        })

    return results
