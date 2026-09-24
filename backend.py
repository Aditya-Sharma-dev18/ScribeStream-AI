"""
backend.py — LangGraph blog generation workflow.

This module exposes a compiled LangGraph application as `app`,
which is imported by app.py (FastAPI server).

Run from command line:
    python backend.py "Your blog topic here"
"""

import operator
import os
import re
import time
import uuid
from pathlib import Path
from typing import Annotated, List, Literal, Optional, TypedDict

import cloudinary
import cloudinary.uploader
import psycopg
import requests
from better_bing_image_downloader import downloader
from dotenv import load_dotenv, find_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send
from psycopg.rows import dict_row
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────
load_dotenv(find_dotenv(), override=True)


# ─────────────────────────────────────────────────────────
# Database URL helper
# ─────────────────────────────────────────────────────────
def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL "
            "External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


# ─────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────
class Task(BaseModel):
    id: int = Field(..., description="Sequential section number (1, 2, 3...)")
    title: str = Field(..., description="Authoritative, descriptive H2 title")
    brief: str = Field(
        ...,
        description=(
            "Comprehensive blueprint of specific mechanisms, concepts, "
            "and nuances to cover"
        ),
    )
    target_word_count: int = Field(
        default=450,
        description="Target word count for depth (typically 400-700 words)",
    )
    key_takeaways: List[str] = Field(
        ...,
        description=(
            "2-4 technical concepts, terms, or practical insights "
            "that must be explained"
        ),
    )
    section_role: Literal[
        "architectural_walkthrough",
        "deep_technical_dive",
        "foundational_concept",
    ] = Field(
        ...,
        description="Structural function of this section.",
    )
    include_code_or_math: bool = Field(
        default=False,
        description=(
            "True if this section requires concrete code blocks, "
            "ASCII diagrams, or math."
        ),
    )


class Plan(BaseModel):
    title: str = Field(
        ...,
        description="High-impact, SEO-optimized title without clickbait fluff",
    )
    target_audience: str = Field(
        ...,
        description=(
            "Target reader persona (e.g., Applied ML Engineers, "
            "Senior Software Engineers)"
        ),
    )
    technical_depth: str = Field(
        ...,
        description="Level: Intermediate, Advanced, or Production-Grade",
    )
    tasks: List[Task] = Field(
        ...,
        description="5-7 logically progressing sections",
    )


class RouterDecision(BaseModel):
    need_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class Evidence_item(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class Evidence_pack(BaseModel):
    evidence: List[Evidence_item] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────
# Graph state
# ─────────────────────────────────────────────────────────
class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[dict, lambda a, b: {**a, **b}]
    final: str
    need_research: bool
    mode: str
    queries: List[str]
    evidence: List[Evidence_item]
    images: Annotated[dict, lambda a, b: {**a, **b}]
    task: Task
    published_url: str


# ─────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────
model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
    num_predict=8192,
)


# ─────────────────────────────────────────────────────────
# Cloudinary
# ─────────────────────────────────────────────────────────
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)


def upload_to_cloudinary(local_path: str, public_id_prefix: str) -> str:
    """Upload local image to Cloudinary and return public HTTPS URL."""
    if not local_path or not os.path.exists(local_path):
        return ""
    try:
        res = cloudinary.uploader.upload(
            local_path,
            folder="scribestream_blogs",
            public_id=public_id_prefix,
            overwrite=True,
            resource_type="image",
        )
        url = res.get("secure_url", "")
        print(f"-> [Cloudinary] Uploaded: {url}")
        return url
    except Exception as e:
        print(f"[Cloudinary Error] {e}")
        return ""


# ─────────────────────────────────────────────────────────
# Router node
# ─────────────────────────────────────────────────────────
ROUTER_SYSTEM = """You route technical topics to a research mode.

Modes:
- closed_book: timeless/conceptual topics. need_research=False, queries=[].
- hybrid: standard concept but needs recent API/framework updates. need_research=True, 2-4 queries.
- open_book: bleeding-edge releases, benchmarks, current events. need_research=True, 2-4 queries.

When need_research=True, queries must be keyword-dense (no "how to", no filler),
and cover implementation + edge-cases + official docs.

Output must match RouterDecision schema exactly.

Examples:
Topic: "Explain quicksort time complexity" -> closed_book, [], False
Topic: "LangChain v0.3 hybrid search with Pinecone" -> hybrid, ["LangChain v0.3 retrieval chain hybrid", "Pinecone sparse dense LangChain", "LangChain 0.3 migration guide"], True
Topic: "DeepSeek-V3 vs Claude 3.5 SWE-bench scores" -> open_book, ["DeepSeek V3 SWE-bench verified", "Claude 3.5 Sonnet SWE-bench leaderboard"], True
""".strip()


def router_node(state: State) -> dict:
    topic = state["topic"]
    decider = model.with_structured_output(RouterDecision, method="json_schema")
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}"),
        ]
    )

    return {
        "need_research": decision.need_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }


def route_next(state: State) -> str:
    return "research" if state.get("need_research", False) else "orchestrator"


# ─────────────────────────────────────────────────────────
# Research node
# ─────────────────────────────────────────────────────────
def tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []
    for result in results or []:
        normalized.append(
            {
                "title": result.get("title") or "",
                "url": result.get("url") or "",
                "snippet": result.get("content") or result.get("snippet") or "",
                "published_at": result.get("published_date")
                or result.get("published_at"),
                "source": result.get("source"),
            }
        )
    return normalized


RESEARCHER_SYSTEM_PROMPT = """Extract dense technical evidence from raw web search results.

For each result worth keeping, produce an Evidence_item with:
- title, url, published_at, source (copied verbatim from the result)
- snippet: an atomic, self-contained technical assertion (exact numbers, API names, tensor shapes, version tags)

Reject: marketing copy, tutorials, forum banter, affiliate links.
Deduplicate by URL. If sources conflict across versions, keep both and note the version.
Do not invent facts not present in the results.
""".strip()


def research_node(state: State) -> dict:
    queries = (state.get("queries", []) or [])[:10]
    max_results = 3

    raw_results: List[dict] = []
    for q in queries:
        raw_results.extend(tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

    trimmed_results = []
    for r in raw_results[:12]:
        trimmed_results.append(
            {
                "title": (r.get("title") or "")[:200],
                "url": r.get("url") or "",
                "snippet": (r.get("snippet") or "")[:500],
                "source": r.get("source"),
            }
        )

    extractor = model.with_structured_output(Evidence_pack, method="json_schema")
    pack: Evidence_pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Topic: {state.get('topic', '')}\n\n"
                    f"Raw results:\n{trimmed_results}"
                )
            ),
        ]
    )

    seen_urls = set()
    deduped_evidence: List[Evidence_item] = []

    for item in pack.evidence:
        clean_url = (item.url or "").strip().rstrip("/").lower()
        if clean_url and clean_url not in seen_urls:
            seen_urls.add(clean_url)
            deduped_evidence.append(item)
        elif not clean_url:
            deduped_evidence.append(item)

    return {"evidence": deduped_evidence}


# ─────────────────────────────────────────────────────────
# Orchestrator node
# ─────────────────────────────────────────────────────────
ORCHESTRATOR_SYSTEM_PROMPT = """You decompose a technical topic into 5-7 section tasks for downstream writers.

Each task must be:
- specific (no "Introduction", no "Conclusion", no "Future Outlook")
- technically deep (mechanisms, trade-offs, tensor shapes, complexities)
- scoped to one subtopic

Use plain ASCII for math: R^(B x L x D), sqrt(d_k), O(L^2). No unicode math symbols.
Do not escape underscores.

Return a Plan matching the schema exactly.
""".strip()


def orchestrator(state: State) -> dict:
    structured_planner = model.with_structured_output(Plan, method="json_schema")
    plan = structured_planner.invoke(
        [
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n\n"
                    "Generate a rigorous, publication-grade blueprint."
                )
            ),
        ]
    )
    return {"plan": plan}


# ─────────────────────────────────────────────────────────
# Fanout
# ─────────────────────────────────────────────────────────
def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task,
                "topic": state["topic"],
                "plan": state["plan"],
                "evidence": state.get("evidence", []),
            },
        )
        for task in state["plan"].tasks
    ]


# ─────────────────────────────────────────────────────────
# Worker node
# ─────────────────────────────────────────────────────────
WORKER_SYSTEM_PROMPT = r"""You are a Principal Software Engineer and Technical Author drafting an authoritative, standalone technical section for a comprehensive engineering manual.

Your writing is deterministic, production-hardened, and unapologetically technical. You do not explain concepts like a teacher to a student; you document systems like an L7 staff engineer writing mission-critical architectural specs.

---
### STRUCTURAL ARCHITECTURE
- Title: Begin immediately with `## {section_title}` on line 1. No preceding text.
- Hierarchy: Subdivide arguments using `###` headings for distinct sub-components or execution phases.
- Density: Use structured Markdown tables for multi-variable trade-offs (e.g., latency vs. throughput vs. memory footprint) and bullet points for runtime invariants.
- No Section Fluff: Absolute prohibition on introductory roadmaps ("In this section...", "Let's dive into...", "We will explore...") or closing summaries ("In conclusion...", "To summarize..."). End abruptly and cleanly on the final concrete architectural invariant or failure mode.

---
### CODE, MATH & SPECIFICATION DIRECTIVES
- Concrete Over Conceptual: Replace abstract nouns with exact types, runtime parameters, hardware constraints, and tensor geometries (e.g., denote shapes explicitly like `(B, S, D)`).
- Algorithmic Rigor: State formal time and space complexity using exact Big-O notation (e.g., $\mathcal{O}(N \cdot D + S^2)$). Enclose inline math in $...$ and standalone equations in $$...$$.
- Runnable Production Code: If illustrating an implementation, output idiomatic, strictly typed, runnable code (Python 3.11+, PyTorch, or target framework). Include edge-case handling and inline dimension comments. Never generate pseudo-code, placeholder ellipses (`# TODO`), or non-functional mocks.
- Failure Modes & Edge Cases: Explicitly document race conditions, memory leaks, OOM triggers, or network partition behavior relevant to the section.

---
### GROUND TRUTH INJECTION
- Ground your section strictly in the provided `task`, `plan`, and retrieved technical evidence.
- Maintain consistent terminology with the overall document architecture.
- Do not repeat high-level context already established in other sections; address only your scoped component with deep technical fidelity.
""".strip()


def worker(payload: dict) -> dict:
    task: Task = payload["task"]
    plan: Plan = payload["plan"]
    topic: str = payload["topic"]

    takeaways_formatted = "\n".join(f"- {item}" for item in task.key_takeaways)

    prompt = (
        f"Document Title: {plan.title}\n"
        f"Target Audience: {plan.target_audience} (Depth: {plan.technical_depth})\n"
        f"Overarching Topic: {topic}\n"
        f"-----------------------------------------\n"
        f"Current Section ID: {task.id}\n"
        f"Current Section Title: {task.title}\n"
        f"Section Role: {task.section_role}\n"
        f"Target Length: ~{task.target_word_count} words\n"
        f"Include Code/Math: {task.include_code_or_math}\n\n"
        f"Detailed Blueprint to Execute:\n{task.brief}\n\n"
        f"Mandatory Inclusions:\n{takeaways_formatted}\n\n"
        "Write only the Markdown content for this section now."
    )

    response = model.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    return {"sections": {task.id: response.content.strip()}}


# ─────────────────────────────────────────────────────────
# Image subgraph
# ─────────────────────────────────────────────────────────
class imagesubgraphstate(TypedDict):
    section_id: int
    prompt: str
    image_path: str


def build_query(prompt: str, section_id: int) -> str:
    """Build a meaningful search query by extracting keywords."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", prompt)
    words = [w for w in cleaned.split() if len(w) > 3]
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "into",
        "are", "was", "were", "have", "has", "had", "but", "not",
        "you", "your", "our", "their", "its", "about", "over", "under",
        "role", "impact", "complexity", "interplay", "consequences",
    }
    keywords = [w for w in words if w.lower() not in stopwords]
    query = " ".join(keywords[:4])
    if not query or len(query) < 5:
        query = f"system architecture diagram {section_id}"
    return f"{query} architecture diagram dark mode"


def is_valid_image(path: str, min_bytes: int = 5000) -> bool:
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size < min_bytes:
            return False
        with open(p, "rb") as f:
            header = f.read(12)
        if header[:3] == b"\xff\xd8\xff":
            return True
        if header[:4] == b"\x89PNG":
            return True
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return True
        if header[:4] == b"GIF8":
            return True
        return False
    except Exception:
        return False


def search_serper_images(query: str, num: int = 3) -> list:
    """Fetch image URLs via Serper API."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        print("  [serper] SERPER_API_KEY missing in .env")
        return []

    try:
        response = requests.post(
            "https://google.serper.dev/images",
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": num},
            timeout=30,
        )
        if response.status_code != 200:
            print(f"  [serper] HTTP {response.status_code}: {response.text[:200]}")
            return []
        data = response.json()
        return [
            img.get("imageUrl")
            for img in data.get("images", [])
            if img.get("imageUrl")
        ]
    except Exception as e:
        print(f"  [serper] error: {e}")
        return []


def download_image(url: str, save_path: str) -> bool:
    """Download image and save to disk."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            ),
        }
        r = requests.get(url, headers=headers, timeout=30, stream=True)
        if r.status_code != 200:
            return False
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return is_valid_image(save_path)
    except Exception:
        return False


def run_bing_generator(prompt: str, section_id: int) -> str:
    image_dir = Path("images")
    image_dir.mkdir(parents=True, exist_ok=True)
    out_dir = image_dir / f"{section_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_query = build_query(prompt, section_id)
    final_query = re.sub(r"[^a-zA-Z0-9\s]", " ", final_query)
    final_query = re.sub(r"\s+", " ", final_query).strip()[:80]

    print(f"  [query] {final_query}")

    # Primary: Serper API
    image_urls = search_serper_images(final_query, num=3)
    if image_urls:
        print(f"  [serper] Got {len(image_urls)} image URLs")
        for i, url in enumerate(image_urls):
            save_path = str(out_dir / f"serper_{i}.jpg")
            if download_image(url, save_path):
                print(f"  [ok] Downloaded: {save_path}")
                cdn_url = upload_to_cloudinary(
                    save_path, f"section_{section_id}"
                )
                if cdn_url:
                    return cdn_url
                else:
                    print("  [warn] Cloudinary failed, trying next URL")
                    continue
    else:
        print("  [warn] No Serper results")

    # Fallback: better-bing-image-downloader
    print("  [fallback] trying better-bing-image-downloader")
    for engine in ["bing", "duckduckgo"]:
        try:
            count = downloader(
                query=final_query,
                limit=1,
                output_dir=str(out_dir),
                engine=engine,
                verbose=False,
                timeout=60,
            )
            if count == 0:
                continue
            files = list(out_dir.glob("**/*.jpg")) + list(
                out_dir.glob("**/*.png")
            )
            for f in files:
                if is_valid_image(str(f)):
                    cdn_url = upload_to_cloudinary(
                        str(f), f"section_{section_id}"
                    )
                    if cdn_url:
                        print(f"  [ok] Fallback via {engine}: {cdn_url}")
                        return cdn_url
        except Exception as e:
            print(f"  [fallback] {engine} error: {e}")
            continue

    print(f"  [fail] All attempts exhausted for section {section_id}")
    return ""


def image_generator_node(state: imagesubgraphstate):
    section_id = state.get("section_id", 0)
    prompt = state.get("prompt", "")
    print(f"generating the image {section_id}")
    path = run_bing_generator(prompt, section_id)
    return {"image_path": path}


subgraph = StateGraph(imagesubgraphstate)
subgraph.add_node("generate_image", image_generator_node)
subgraph.add_edge(START, "generate_image")
subgraph.add_edge("generate_image", END)
image_subgraph = subgraph.compile()


# ─────────────────────────────────────────────────────────
# Diagram orchestrator
# ─────────────────────────────────────────────────────────
def diagram_orchestrator_node(state: State) -> dict:
    plan: Plan = state["plan"]
    topic: str = state["topic"]
    collected_images = {}

    visual_roles = [
        "architectural_walkthrough",
        "deep_technical_dive",
        "foundational_concept",
    ]

    for task in plan.tasks:
        if task.section_role in visual_roles or task.include_code_or_math:
            sub_res = image_subgraph.invoke(
                {
                    "section_id": task.id,
                    "prompt": f"{task.title} architecture mechanics for {topic}",
                    "image_path": "",
                }
            )
            if sub_res.get("image_path"):
                collected_images[task.id] = sub_res["image_path"]

    return {"images": collected_images}


# ─────────────────────────────────────────────────────────
# Reducer
# ─────────────────────────────────────────────────────────
def reducer(state: State) -> dict:
    plan: Plan = state["plan"]
    title = getattr(plan, "title", "Technical Blog Post")
    images = state.get("images", {})

    meta_header = (
        f"# {title}\n\n"
        f"> **Audience:** {getattr(plan, 'target_audience', 'Engineers')} | "
        f"**Level:** {getattr(plan, 'technical_depth', 'Advanced')}\n\n"
    )

    processed_sections = []
    sections_map = state.get("sections", {})
    for idx in sorted(sections_map.keys()):
        sec_text = sections_map[idx]
        if idx in images:
            img_tag = (
                f"\n\n![Figure {idx}]({images[idx]})\n"
                f"*<p align='center'>Figure {idx}: System Architecture</p>*\n"
            )
            sec_text = f"{sec_text}{img_tag}"
        processed_sections.append(sec_text)

    body = "\n\n---\n\n".join(processed_sections).strip()
    final_md = f"{meta_header}\n{body}\n"

    clean_title = (
        re.sub(r'[\\/*?:"<>|]', "", title).strip().lower().replace(" ", "_")
    )
    filename = f"{clean_title}.md"

    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")

    return {"final": final_md}


# ─────────────────────────────────────────────────────────
# Publisher
# ─────────────────────────────────────────────────────────
def post_to_devto(
    title: str,
    markdown_content: str,
    tags: list | None = None,
    publish_live: bool = False,
) -> str:
    api_key = os.getenv("DEV_TO_API_KEY")
    if not api_key:
        print("[Dev.to Warning] DEV_TO_API_KEY is missing in .env file")
        return ""

    response = requests.post(
        "https://dev.to/api/articles",
        headers={
            "api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "article": {
                "title": title,
                "body_markdown": markdown_content,
                "published": publish_live,
                "tags": tags or ["ai", "python", "machinelearning"],
            }
        },
        timeout=30,
    )

    if not response.ok:
        print(f"Dev.to publishing failed: {response.status_code} {response.text}")
        return ""

    article_url = response.json().get("url", "")
    print(f"Successfully posted to DEV.to: {article_url}")
    return article_url


def publisher_node(state: State) -> dict:
    plan: Plan = state["plan"]
    title = getattr(plan, "title", "Technical Deep Dive")
    content = state.get("final", "")

    print("Publishing blog to DEV.to...")
    article_link = post_to_devto(
        title=title,
        markdown_content=content,
        tags=["python", "ai", "machinelearning"],
        publish_live=False,  # Set True to publish live directly
    )

    return {"published_url": article_link}


# ─────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────
graph = StateGraph(State)

retry = RetryPolicy(max_attempts=3)

graph.add_node("router", router_node, retry_policy=retry)
graph.add_node("research", research_node, retry_policy=retry)
graph.add_node("orchestrator", orchestrator, retry_policy=retry)
graph.add_node(
    "diagram_orchestrator",
    diagram_orchestrator_node,
    retry_policy=retry,
)
graph.add_node("worker", worker, retry_policy=retry)
graph.add_node("reducer", reducer)
graph.add_node("publisher", publisher_node)

graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router",
    route_next,
    {"research": "research", "orchestrator": "orchestrator"},
)

graph.add_edge("research", "orchestrator")
graph.add_edge("orchestrator", "diagram_orchestrator")
graph.add_conditional_edges(
    "diagram_orchestrator", fanout, {"worker": "worker"}
)
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", "publisher")
graph.add_edge("publisher", END)


# ─────────────────────────────────────────────────────────
# Checkpointer + compile
# ─────────────────────────────────────────────────────────
DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

# Exposed as `app` so app.py can import it directly:
#     from backend import app as workflow
app = graph.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    default_topic = (
        "The AI Frontend Paradox: Why AI Coding Assistants Are "
        "Creating a Lost Decade for Developers"
    )
    topic = sys.argv[1] if len(sys.argv) > 1 else default_topic

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    out = app.invoke(
        {
            "topic": topic,
            "sections": {},
        },
        config=config,
    )

    print(f"\nDone! Character length: {len(out['final'])}")
    if out.get("published_url"):
        print(f"Dev.to Link: {out['published_url']}")