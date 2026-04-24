# 🤖 LLM Agentic Chatbot
### Multi-Agent Orchestration · LangGraph · LangChain · Groq · Streamlit · Tavily

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-4B8BBE?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3A4A?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-LLM%20Inference-F55036?logoColor=white)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tavily](https://img.shields.io/badge/Tavily-Search-00C7B7?logoColor=white)](https://tavily.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 Overview

This project is a **production-structured Agentic AI Chatbot** built with **LangGraph** and **LangChain**, powered by the **Groq LLM** provider for ultra-fast inference. It demonstrates how to build stateful, multi-agent systems that go beyond a simple chatbot — using graph-based workflows, real-time web search, and an automated AI news summarization pipeline.

The application exposes **three distinct use cases**, each backed by its own compiled LangGraph state graph, switchable through a Streamlit sidebar at runtime — no code changes needed.

| Use Case | Description |
|----------|-------------|
| 🗣️ **Basic Chatbot** | Conversational LLM with stateful message history |
| 🌐 **Chatbot With Web Search** | ReAct agent that fetches live web data via Tavily Search |
| 📰 **AI News Summarizer** | Multi-node pipeline: fetch → summarize → save to Markdown |

---

## ✨ Features

- **Stateful Conversations** — LangGraph's `add_messages` reducer accumulates the full message history across turns, giving the model genuine conversational memory without an external store.
- **ReAct Agent Loop** — The web-search mode implements a real ReAct (Reason + Act) cycle: the LLM decides *when* to call tools, executes them, and loops back to refine its answer until it's done.
- **Modular Graph Architecture** — Each use case compiles its own `StateGraph`. Adding a new use case means adding a new graph builder method — nothing else changes.
- **AI News Pipeline** — A sequential 3-node LangGraph pipeline fetches top AI news globally and in India, summarizes it with an LLM into date-sorted Markdown, and persists it to disk automatically.
- **Groq Inference** — Ultra-low-latency LLM calls via `langchain_groq`, with model selection exposed to the user at runtime via the sidebar.
- **Streamlit UI** — API key entry and model/use-case selection in the sidebar; clean chat interface for all three modes.

---

## 🏗️ Architecture

### High-Level System Design

```
┌──────────────────────────────────────────────────────────────┐
│                        Streamlit UI                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Sidebar: GROQ_API_KEY · TAVILY_API_KEY                │  │
│  │           Model Selection · Use Case Switch            │  │
│  └─────────────────────────┬──────────────────────────────┘  │
│                            │                                 │
│                            ▼                                 │
│                  ┌──────────────────┐                        │
│                  │   GroqLLM Layer  │   ChatGroq wrapper     │
│                  └────────┬─────────┘                        │
│                           │                                  │
│                           ▼                                  │
│                  ┌──────────────────┐                        │
│                  │   GraphBuilder   │   setup_graph(usecase) │
│                  └────────┬─────────┘                        │
│                           │                                  │
│          ┌────────────────┼─────────────────┐                │
│          ▼                ▼                 ▼                │
│   [Basic Graph]   [Tools Graph]     [AI News Graph]          │
└──────────────────────────────────────────────────────────────┘
```

---

### Graph 1 — Basic Chatbot

```
START ──► chatbot ──► END
```

The `BasicChatbotNode` invokes the LLM with the full accumulated message list and returns the response. LangGraph's `add_messages` reducer ensures each new response is appended rather than overwriting history.

---

### Graph 2 — Chatbot With Web Search (ReAct Loop)

```
                  ┌─────────────────────────────┐
                  │                             │
START ──► chatbot ──► tools_condition ──► tools ┘
               │
               └──► END  (when no tool call needed)
```

- `ChatbotWithToolNode` binds `TavilySearchResults` to the LLM using `llm.bind_tools(tools)`.
- `tools_condition` is LangGraph's built-in conditional edge: if the LLM emits a `ToolCall` message, it routes to `tools`; otherwise it routes to `END`.
- `ToolNode` executes the Tavily search and injects results back into the message state, completing the ReAct cycle.

---

### Graph 3 — AI News Summarizer (Sequential Pipeline)

```
START ──► fetch_news ──► summarize_news ──► save_result ──► END
```

| Node | What It Does |
|------|-------------|
| `fetch_news` | Reads frequency (`daily/weekly/monthly/year`) from state, calls Tavily News API, stores up to 20 results |
| `summarize_news` | Formats articles via `ChatPromptTemplate`, calls LLM, stores date-sorted Markdown summary |
| `save_result` | Writes summary to `./AINews/{frequency}_summary.md` |

---

## 📂 Project Structure

```
LLM_agentic_chatbot/
│
├── langgraphagenticai/
│   │
│   ├── LLM/
│   │   ├── __init__.py
│   │   └── groqllm.py                    # GroqLLM wrapper — reads API key & model from UI
│   │
│   ├── graph/
│   │   └── graph_build.py                # GraphBuilder — compiles StateGraph per use case
│   │
│   ├── nodes/
│   │   ├── basic_chatbot_nodes.py        # BasicChatbotNode — simple LLM invoke
│   │   ├── chatbot_with_tools node.py    # ChatbotWithToolNode — LLM + bind_tools + ReAct
│   │   └── ainews_node.py               # AINewsNode — fetch / summarize / save pipeline
│   │
│   ├── tools/
│   │   └── searchtool.py                # TavilySearchResults tool + ToolNode factory
│   │
│   └── state.py                         # Shared LangGraph State (TypedDict + add_messages)
│
├── AINews/                              # Auto-created output directory for news summaries
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🔬 Code Deep Dive

### State Definition — `state.py`

```python
from typing_extensions import TypedDict, List
from langgraph.graph.message import add_messages
from typing import Annotated

class State(TypedDict):
    messages: Annotated[List, add_messages]
```

`add_messages` is a LangGraph **reducer** — instead of overwriting `messages` on each node step, it appends new messages to the existing list. This single annotation is what gives the chatbot persistent conversational memory across turns, with no external database required.

---

### LLM Layer — `LLM/groqllm.py`

```python
class GroqLLM:
    def __init__(self, user_controls_input):
        self.user_controls_input = user_controls_input

    def get_llm_model(self):
        groq_api_key = str(self.user_controls_input["GROQ_API_KEY"])
        selected_groq_model = self.user_controls_input["selected_groq_model"]
        llm = ChatGroq(api_key=groq_api_key, model=selected_groq_model)
        return llm
```

The `user_controls_input` dict is populated directly from the Streamlit sidebar, making both the API key and model swappable at runtime without touching any code.

---

### Graph Builder — `graph/graph_build.py`

```python
class GraphBuilder:
    def __init__(self, model):
        self.llm = model
        self.graph_builder = StateGraph(State)

    def setup_graph(self, usecase: str):
        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph()
        if usecase == "Chatbot With Web":
            self.chatbot_with_tools_build_graph()
        if usecase == "AI News":
            self.ai_news_builder_graph()
        return self.graph_builder.compile()
```

Each use case calls its own dedicated builder method that wires nodes and edges, then `compile()` returns a fully executable `CompiledGraph`. The single `setup_graph` entrypoint keeps the Streamlit layer clean and use cases fully isolated from each other.

---

### ReAct Tool Node — `chatbot_with_tools node.py`

```python
def create_chatbot(self, tools):
    llm_with_tools = self.llm.bind_tools(tools)

    def chatbot_node(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    return chatbot_node
```

`bind_tools` injects the tool schemas into the LLM's context. When the model decides a web search is needed, it emits a `ToolCall` message object instead of a regular text reply. LangGraph's `tools_condition` detects this and routes execution to the `ToolNode` automatically — this is the core of the ReAct loop.

---

### AI News Node — `nodes/ainews_node.py`

```python
# fetch_news — maps frequency to Tavily time range
time_range_map = {'daily': 'd', 'weekly': 'w', 'monthly': 'm', 'year': 'y'}
response = self.tavily.search(
    query="Top AI news in India and Globally",
    topic="news",
    time_range=time_range_map[frequency],
    include_answer="advanced",
    max_results=20
)

# summarize_news — structured Markdown prompt
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """Summarize AI news articles into markdown format.
    - Date in **YYYY-MM-DD** format in IST timezone
    - Concise summary from latest news
    - Sort by date (latest first)
    - Source URL as clickable link
    Format: ### [Date] \n - [Summary](URL)"""),
    ("user", "Articles:\n{articles}")
])

# save_result — persists to disk
filename = f"./AINews/{frequency}_summary.md"
with open(filename, 'w') as f:
    f.write(summary)
```

The structured prompt enforces a consistent, readable Markdown output format across all LLM responses — making saved files immediately shareable.

---

### Search Tool — `tools/searchtool.py`

```python
def get_tools():
    tools = [TavilySearchResults(max_results=2)]
    return tools

def create_tool_node(tools):
    return ToolNode(tools=tools)
```

`TavilySearchResults` is a LangChain-wrapped tool for real-time web search. `max_results=2` keeps the injected context concise and latency low. `ToolNode` is LangGraph's prebuilt executor — it handles tool invocation, result formatting, and automatic injection back into the message state.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A [Groq API Key](https://console.groq.com/) (free tier available)
- A [Tavily API Key](https://app.tavily.com/) (required for web search and AI News modes)

### 1. Clone the Repository

```bash
git clone https://github.com/akshra09/LLM_agentic_chatbot.git
cd LLM_agentic_chatbot
```

### 2. Create & Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Set them as environment variables (recommended) or enter them directly in the Streamlit sidebar:

```bash
export GROQ_API_KEY="gsk_your_groq_key_here"
export TAVILY_API_KEY="tvly_your_tavily_key_here"
```

### 5. Create the Output Directory

```bash
mkdir AINews
```

### 6. Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🖥️ Usage Guide

### 🗣️ Basic Chatbot

1. Select **"Basic Chatbot"** in the sidebar
2. Enter your Groq API Key and select a model (e.g. `llama3-70b-8192`)
3. Start chatting — the model maintains full conversational context across turns

### 🌐 Chatbot With Web Search

1. Select **"Chatbot With Web"** in the sidebar
2. Enter both your Groq and Tavily API Keys
3. Ask anything requiring real-time information — e.g. *"What happened in AI this week?"*
4. The agent will autonomously decide whether to search, execute the search, and synthesize results

### 📰 AI News Summarizer

1. Select **"AI News"** in the sidebar
2. Type one of: `daily`, `weekly`, `monthly`, or `year`
3. The pipeline automatically:
   - Fetches top 20 AI news results from Tavily
   - Summarizes them into date-sorted Markdown via the LLM
   - Saves the output to `./AINews/{frequency}_summary.md`

---

## ⚙️ Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | Core LLM framework and abstractions |
| `langgraph` | Graph-based agent orchestration and state management |
| `langchain_community` | TavilySearchResults and community tool integrations |
| `langchain_core` | Prompts, messages, and base abstractions |
| `langchain_groq` | Groq LLM provider integration |
| `langchain_openai` | OpenAI-compatible LLM support |
| `faiss-cpu` | Vector store (ready for RAG extension) |
| `streamlit` | Web UI framework |
| `tavily-python` | Tavily Search and News API client |

---

## 🧩 Extending the Project

The modular `GraphBuilder` makes adding new agentic use cases straightforward:

**Step 1** — Create a node class in `langgraphagenticai/nodes/`:
```python
class MyCustomNode:
    def __init__(self, llm):
        self.llm = llm
    def process(self, state: State) -> dict:
        # your logic here
        return {"messages": [...]}
```

**Step 2** — Add a builder method in `graph/graph_build.py`:
```python
def my_custom_build_graph(self):
    node = MyCustomNode(self.llm)
    self.graph_builder.add_node("my_node", node.process)
    self.graph_builder.add_edge(START, "my_node")
    self.graph_builder.add_edge("my_node", END)
```

**Step 3** — Register it in `setup_graph`:
```python
if usecase == "My Custom Mode":
    self.my_custom_build_graph()
```

**Step 4** — Add it to the Streamlit sidebar use case list.

### Ideas for Extension

- 📄 **RAG over documents** — `faiss-cpu` is already installed; add a retrieval node that chunks and embeds uploaded files
- 🧑‍💼 **Multi-agent supervisor** — use a supervisor node to route queries to specialized sub-agents (coding, math, research)
- 🗃️ **Persistent memory** — replace in-memory state with LangGraph's `SqliteSaver` or `PostgresSaver` checkpointers for cross-session memory
- 🔔 **Scheduled news delivery** — wrap the AI News pipeline in a cron job and send summaries via email or Slack

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `GROQ_API_KEY` error on launch | Set it as an env variable or enter it in the sidebar |
| Tavily returns no results | Verify your `TAVILY_API_KEY` is valid and has remaining quota |
| `FileNotFoundError` for AINews | Run `mkdir AINews` in the project root |
| `ModuleNotFoundError` | Make sure your virtual environment is active and `pip install -r requirements.txt` was run |
| Streamlit app won't start | Ensure you're running `streamlit run app.py` from the project root directory |
| LLM returns no tool calls in web mode | Try a larger Groq model like `llama3-70b-8192` — smaller models may not trigger tools reliably |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [LangGraph](https://langchain-ai.github.io/langgraph/) — graph-based agent orchestration
- [LangChain](https://www.langchain.com/) — LLM toolkit and integrations
- [Groq](https://groq.com/) — blazing-fast LLM inference
- [Tavily](https://tavily.com/) — search API purpose-built for LLM agents
- [Streamlit](https://streamlit.io/) — rapid ML app UI development

---


