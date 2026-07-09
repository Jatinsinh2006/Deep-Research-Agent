# 🔬 Deep Research Agent

An AI-powered research assistant that breaks down any topic into focused sub-questions, researches each one, and synthesizes a complete report — powered by **Groq's Llama 3.3 70B** for blazing-fast, free inference.

Available in two forms:
- 🖥️ **CLI** (`research_agent.py`) — terminal-based, interactive chat
- 🌐 **Web App** (`streamlit_app.py`) — clean UI with progress tracking and downloadable reports

## ✨ Features

- ⚡ **Quick Answer mode** — direct, concise answers
- 🔬 **Deep Research mode** — multi-step pipeline:
  1. Generates sub-questions on the topic
  2. Researches each sub-question
  3. Synthesizes everything into one structured report
- 💾 Auto-saves reports as JSON + Markdown
- 📥 One-click report download (web app)

## 🛠️ Tech Stack

| Component | Tech |
|---|---|
| Language | Python 3.13 |
| AI Model | Llama 3.3 70B via Groq API |
| Web UI | Streamlit |
| API Client | OpenAI SDK (Groq's OpenAI-compatible endpoint) |
| Config | python-dotenv |

## 🚀 Setup

1. Clone the repo:
```bash
   git clone https://github.com/your-username/deep-research-agent.git
   cd deep-research-agent
```

2. Create a virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Get a **free** Groq API key at [console.groq.com](https://console.groq.com)

5. Create a `.env` file: