[README.md](https://github.com/user-attachments/files/32608078/README.md)
# ✍️ ScribeStream AI — Autonomous Technical Blog Generator

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-purple.svg)](https://ollama.com/)
[![Qwen](https://img.shields.io/badge/Qwen3-1.7B-teal.svg)](https://ollama.com/library/qwen3)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Checkpointer-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#-license)

> **Transform any technical topic into a publication-ready blog post using a multi-agent LangGraph pipeline, local LLMs, and automated research + image generation.**

---

## 📋 **Table of Contents**

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Configuration](#-configuration)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🚨 **Problem Statement**

### **The Challenge**

Writing high-quality technical blog posts is time-consuming and demands deep expertise:

- **Research overhead**: Gathering authoritative sources, benchmarks, and version-specific details
- **Structural planning**: Breaking a topic into a coherent, non-repetitive narrative
- **Visual assets**: Creating or sourcing relevant diagrams for each section
- **Publishing friction**: Manually formatting, uploading images, and posting to platforms
- **Cost barrier**: Cloud LLM APIs rack up significant bills for content generation

### **The Gap**

There is no **open, offline-capable, end-to-end pipeline** that:

- Researches a topic automatically
- Plans the article structure with technical depth
- Writes each section in a consistent engineering voice
- Sources and uploads relevant images
- Publishes directly to a blogging platform

---

## 💡 **Solution Overview**

**ScribeStream AI** is a **multi-agent LangGraph workflow** that solves this end-to-end:

1. **Router** — Decides whether the topic needs live web research
2. **Research** — Fetches and deduplicates authoritative sources via Tavily
3. **Orchestrator** — Produces a structured 5-7 section `Plan` with technical depth
4. **Diagram Orchestrator** — Generates one relevant image per section via Serper + Cloudinary
5. **Workers (parallel)** — Write each section with production-grade technical fidelity
6. **Reducer** — Merges sections + images into a single Markdown article
7. **Publisher** — Posts to Dev.to as a draft or live article

### **What Makes It Special**

- ✅ **Runs locally** on `qwen3:1.7b` via Ollama — **zero LLM API cost**
- ✅ **Full pipeline** — research → plan → write → illustrate → publish
- ✅ **Structured outputs** via Pydantic schemas (`Plan`, `Task`, `RouterDecision`, `Evidence_pack`)
- ✅ **PostgreSQL checkpointer** — resumable runs, crash-safe
- ✅ **FastAPI + SSE web UI** — watch every stage live in the browser
- ✅ **Plug-and-play publishers** — Dev.to today, WordPress/Medium tomorrow

---

## ✨ **Features**

### 🌟 **Core Features**

| Feature | Description |
|---------|-------------|
| **🧭 Smart Router** | Classifies topic into `closed_book`, `hybrid`, or `open_book` mode |
| **🔍 Auto Research** | Tavily web search + LLM-powered evidence extraction |
| **📐 Structured Planning** | Pydantic-validated `Plan` with typed `Task` objects |
| **✍️ Parallel Writers** | LangGraph `Send` fanout for concurrent section writing |
| **🖼️ Image Pipeline** | Serper API for search + Cloudinary for CDN hosting |
| **📤 Auto Publish** | Dev.to REST API integration with draft/live toggle |
| **♻️ Resumable** | Postgres-backed checkpointer preserves state on crash |

### 🚀 **Advanced Features**

| Feature | Description |
|---------|-------------|
| **🧠 Local LLM** | `qwen3:1.7b` runs 100% on GPU (even 4GB VRAM) |
| **⚡ SSE Streaming** | Real-time stage updates to the browser |
| **🌐 FastAPI Server** | REST + streaming endpoints for programmatic use |
| **🎨 Premium UI** | Dark theme, live progress ring, timeline, task cards |
| **🖼️ Image Fallback** | Bing/DuckDuckGo fallback if Serper fails |
| **🛡️ Secret-safe** | `.env` based config, `.gitignore` protects keys |
| **📥 Markdown Download** | Generated `.md` files served via API |

### 🎯 **Use Cases**

- **Engineering Blogs**: Generate deep-dive technical posts on autopilot
- **DevRel Teams**: Ship weekly content without hiring writers
- **Personal Branding**: Build a Dev.to presence with consistent quality
- **Learning Tool**: Study how multi-agent pipelines decompose writing
- **Offline Environments**: Run the entire pipeline without cloud LLM APIs

---

## 🛠️ **Tech Stack**

### **Backend**

```mermaid
graph LR
    A[FastAPI] --> B[LangGraph]
    B --> C[Ollama / Qwen3]
    B --> D[Postgres Checkpointer]
    B --> E[Tavily Search]
    B --> F[Serper Images]
    B --> G[Cloudinary CDN]
    B --> H[Dev.to API]
```

| Technology | Purpose |
|------------|---------|
| **FastAPI** | High-performance async API framework |
| **LangGraph** | Multi-agent orchestration with state machines |
| **Ollama** | Local LLM runtime (Qwen3 1.7B) |
| **LangChain** | LLM abstraction, structured output, tools |
| **Pydantic** | Schema validation for `Plan`, `Task`, `Evidence` |
| **Tavily** | Web search for research node |
| **Serper** | Google Images API for visuals |
| **Cloudinary** | Image hosting and CDN delivery |
| **PostgreSQL** | Durable checkpoint store |
| **psycopg** | PostgreSQL driver |

### **Frontend**

| Technology | Purpose |
|------------|---------|
| **HTML5 + Jinja2** | Server-rendered template |
| **CSS3** | Custom design system, dark theme |
| **Vanilla JS** | SSE streaming, DOM updates |
| **Marked.js** | Markdown → HTML rendering |
| **DOMPurify** | XSS-safe HTML sanitization |

### **DevOps & Security**

| Technology | Purpose |
|------------|---------|
| **Python 3.11** | Core language |
| **Uvicorn** | ASGI server |
| **python-dotenv** | Secure credential loading |
| **.gitignore** | Prevents `.env` and generated files from being committed |

---

## 🏗️ **Architecture**

```mermaid
flowchart TD
    A[User submits topic] --> B[Router]
    B -->|needs research| C[Research Node]
    B -->|closed book| D[Orchestrator]
    C --> D
    D --> E[Diagram Orchestrator]
    E --> F[Fanout: Workers]
    F --> G[Worker 1]
    F --> H[Worker 2]
    F --> I[Worker N]
    G --> J[Reducer]
    H --> J
    I --> J
    J --> K[Publisher]
    K --> L[Dev.to Article]
```

### **Data Flow**

1. **Router** classifies the topic and decides if research is needed
2. **Research** runs Tavily queries, extracts evidence, deduplicates by URL
3. **Orchestrator** produces a `Plan` with 5-7 typed `Task` objects
4. **Diagram Orchestrator** generates one image per visual section
5. **Workers** fan out in parallel, one per `Task`, writing Markdown
6. **Reducer** merges sections + images into the final article
7. **Publisher** posts to Dev.to via REST API

### **Graph Topology**

```
START → router → [research | orchestrator]
              → orchestrator → diagram_orchestrator
              → fanout → worker (×N, parallel)
              → reducer → publisher → END
```

---

## 📦 **Installation**

### **Prerequisites**

- **Python 3.11+**
- **Ollama** installed and running ([ollama.com](https://ollama.com))
- **PostgreSQL** database (local or Render/Railway)
- **API Keys**:
  - Tavily (research)
  - Serper (image search)
  - Cloudinary (image hosting)
  - Dev.to (publishing)

### **1. Clone the Repository**

```bash
git clone https://github.com/Aditya-Sharma-dev18/ScribeStream-AI.git
cd ScribeStream-AI
```

### **2. Create Conda Environment**

```bash
conda create -n scribestream python=3.11 -y
conda activate scribestream
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```bash
pip install fastapi uvicorn jinja2 python-dotenv
pip install langgraph langchain-core langchain-ollama langchain-community
pip install pydantic psycopg[binary]
pip install cloudinary requests better-bing-image-downloader
```

### **4. Pull the Local LLM**

```bash
ollama pull qwen3:1.7b
```

Verify GPU offload:

```bash
ollama ps
# Expected: qwen3:1.7b    1.7 GB    100% GPU
```

### **5. Configure Environment**

```bash
cp .env.example .env
```

Edit `.env` with your credentials (see [Configuration](#-configuration)).

### **6. Run the Server**

```bash
uvicorn app:app --reload --port 8000
```

Open **http://127.0.0.1:8000** in your browser.

---

## 🎯 **Usage**

### **Quick Start (Web UI)**

1. Open **http://127.0.0.1:8000**
2. Enter a technical topic, e.g.:
   ```
   The AI Frontend Paradox: Why AI Coding Assistants Are Creating a Lost Decade for Developers
   ```
3. Click **Run agent**
4. Watch the live timeline:
   - Router → Research → Orchestrator → Workers → Reducer → Publisher
5. Read the final article in the result panel
6. Download the `.md` or open the Dev.to draft

### **CLI Usage**

```bash
python backend.py "Your technical topic here"
```

Output:
```
generating the image 1
  [serper] Got 3 image URLs
  [ok] Downloaded: images/1/serper_0.jpg
-> [Cloudinary] Uploaded: https://res.cloudinary.com/.../section_1.png
...
Publishing blog to DEV.to...
Done! Character length: 22223
Dev.to Link: https://dev.to/...
```

### **Example Topics**

- "The AI Frontend Paradox: Why AI Coding Assistants Are Creating a Lost Decade"
- "Design a production-ready RAG system with LangGraph and PostgreSQL"
- "Explain LangGraph subgraphs with a practical multi-agent example"
- "Latest practical developments in open-source AI agents"

---

## 📡 **API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI home page |
| `/api/health` | GET | Health check (`{"status": "ok"}`) |
| `/api/run` | POST | Start a workflow run (SSE stream) |
| `/api/runs/{run_id}/download` | GET | Download generated Markdown |
| `/static/*` | GET | Static assets (CSS, JS) |
| `/images/*` | GET | Generated images |

### **Example: Start a Run (SSE)**

```bash
curl -N -X POST http://127.0.0.1:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"topic": "Explain LangGraph subgraphs"}'
```

Response streams Server-Sent Events:

```
data: {"type":"run_started","run_id":"abc123","topic":"..."}

data: {"type":"stage","id":"router","label":"Analyze the request","status":"running"}

data: {"type":"routing","mode":"hybrid","needs_research":true,"queries":[...]}

data: {"type":"plan","plan":{"title":"...","tasks":[...]}}

data: {"type":"section_complete","task_id":1,"title":"...","markdown":"..."}

data: {"type":"final","markdown":"...","download_url":"/api/runs/abc123/download"}

data: {"type":"done"}
```

---

## 🔧 **Configuration**

### **`.env` Example**

```env
# ── Local LLM ──────────────────────────────────────────
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_BASE_URL=http://localhost:11434

# ── PostgreSQL Checkpointer ───────────────────────────
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# ── Research ──────────────────────────────────────────
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx

# ── Image Search ──────────────────────────────────────
SERPER_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ── Image Hosting ─────────────────────────────────────
CLOUDINARY_CLOUD_NAME=xxxxxxxx
CLOUDINARY_API_KEY=xxxxxxxxxxxxx
CLOUDINARY_API_SECRET=xxxxxxxxxxxxxxxxxxxxxx

# ── Publishing ────────────────────────────────────────
DEV_TO_API_KEY=xxxxxxxxxxxxxxxxxxxx
```

### **Tuning the Model**

In `backend.py`:

```python
model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
    num_predict=8192,
)
```

**Hardware guidance:**

| VRAM | Recommended Model | Notes |
|------|-------------------|-------|
| 4 GB | `qwen3:1.7b` | 100% GPU, fastest |
| 6-8 GB | `qwen3:4b` | Better quality |
| 12 GB+ | `qwen3:8b` or `qwen3.5:9b` | Production-grade output |

### **Publishing Mode**

In `backend.py`, `publisher_node`:

```python
publish_live=False  # Draft (recommended)
publish_live=True   # Live publish
```

---

## 🚀 **Deployment**

### **Option 1: Render.com**

```yaml
# render.yaml
services:
  - type: web
    name: scribestream-ai
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: DATABASE_URL
        sync: false
      - key: TAVILY_API_KEY
        sync: false
      - key: SERPER_API_KEY
        sync: false
      - key: CLOUDINARY_CLOUD_NAME
        sync: false
      - key: CLOUDINARY_API_KEY
        sync: false
      - key: CLOUDINARY_API_SECRET
        sync: false
      - key: DEV_TO_API_KEY
        sync: false
```

### **Option 2: Docker**

```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

> **Note**: Ollama must run on the host machine or a separate GPU-enabled service. The Docker container calls it via `OLLAMA_BASE_URL`.

### **Option 3: Local + Cloud Hybrid**

- **Ollama** runs on your laptop (GPU)
- **FastAPI** runs on Render/Railway
- Set `OLLAMA_BASE_URL` to a public tunnel (e.g., ngrok, Cloudflare Tunnel)

---

## 🔧 **Troubleshooting**

### **Common Issues**

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: psycopg` | `conda activate scribestream` before running |
| `psycopg.OperationalError` | Verify `DATABASE_URL` and `sslmode=require` |
| Ollama shows `73%/27% CPU/GPU` | Model too large for VRAM — switch to `qwen3:1.7b` |
| `[Cloudinary Error] Invalid cloud_name` | Cloud name must be lowercase (e.g., `xwkkdva5`, not `Scribe`) |
| `[serper] HTTP 401` | Invalid Serper API key |
| `[serper] HTTP 429` | Free tier limit reached (2,500/month) |
| `WinError 123` (image download) | Prompt contains invalid Windows path chars — sanitized in code |
| `Could not download image` | Bing/DuckDuckGo rate limit — Serper fallback handles it |
| SSE stream closes early | Check browser console for JS errors |
| Frontend shows "Server offline" | `uvicorn` not running or wrong port |

### **Performance Tips**

- **Use `qwen3:1.7b`** on 4GB VRAM for full GPU offload
- **Set `num_ctx=2048`** in `ChatOllama` to fit KV cache
- **Cache Serper results** by topic to reduce API calls
- **Skip images** during dev (`return {"images": {}}` in `diagram_orchestrator_node`)

---

## 🤝 **Contributing**

### **How to Contribute**

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m "Add amazing feature"`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **Guidelines**

- Follow **PEP 8** for Python code
- Write **docstrings** for all functions
- Add **type hints** where possible
- **Never commit** `.env` or API keys
- Update **README** for new features
- Add **screenshots** for UI changes

---

## 📄 **License**

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Ollama** for making local LLMs accessible
- **LangGraph** for the multi-agent orchestration framework
- **Tavily** for fast, high-quality web search
- **Serper** for reliable Google Images access
- **Cloudinary** for generous image hosting
- **Dev.to** for a developer-friendly publishing API
- **Qwen** team for a small but capable LLM

---

## 🏆 **Project Status**

![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen.svg)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)
![Local_LLM](https://img.shields.io/badge/Local_LLM-100%25_GPU-blue.svg)
![Cost](https://img.shields.io/badge/LLM_Cost-%240.00-success.svg)

---

## 📞 **Contact & Support**

- **GitHub Issues**: [Report a bug](https://github.com/Aditya-Sharma-dev18/ScribeStream-AI/issues)
- **Email**: [sharma.adityaaa0001@gmail.com](mailto:sharma.adityaaa0001@gmail.com)
- **Dev.to**: [@aditya_sharma_f6d5284c3c2](https://dev.to/aditya_sharma_f6d5284c3c2)

---

## ⭐ **Star Us!**

If you find this project useful, please give it a **star ⭐** on GitHub!

---

**Made with ❤️ by Aditya Sharma**
