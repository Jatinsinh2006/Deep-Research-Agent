"""
Deep Research Agent - Streamlit Web App
-----------------------------------------
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide"
)

# ---------- Custom Styling ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        text-align: center;
        color: #F97316;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .report-box {
        background-color: #f7f7f8;
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #F97316;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Agent Logic ----------
class DeepResearchAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    def _ask(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def generate_questions(self, topic: str, num_questions: int = 5) -> list:
        prompt = f"""You are a research planner. Break down this topic into
{num_questions} focused, non-overlapping research questions.

Topic: {topic}

Return ONLY the questions, one per line, no numbering, no extra text."""
        result = self._ask(prompt, max_tokens=400)
        questions = [q.strip("- ").strip() for q in result.split("\n") if q.strip()]
        return questions[:num_questions]

    def research_question(self, question: str) -> str:
        prompt = f"""Answer this research question thoroughly and factually,
in 3-5 sentences. Be specific and avoid filler.

Question: {question}"""
        return self._ask(prompt, max_tokens=500)

    def synthesize_report(self, topic: str, qa_pairs: list) -> str:
        context = "\n\n".join(f"Q: {qa['question']}\nA: {qa['answer']}" for qa in qa_pairs)
        prompt = f"""Using the research findings below, write a clear,
well-structured report on the topic. Include a short summary, key findings,
and a brief conclusion. Do not repeat the raw Q&A format - synthesize it
into flowing sections with markdown headers.

Topic: {topic}

Research Findings:
{context}

Write the final report:"""
        return self._ask(prompt, max_tokens=1500)

    def quick_answer(self, question: str) -> str:
        prompt = f"""Answer this question directly and concisely,
in 2-3 sentences maximum. No preamble.

Question: {question}"""
        return self._ask(prompt, max_tokens=200)


# ---------- Sidebar ----------
api_key = os.getenv("GROQ_API_KEY", "")

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    mode = st.radio("Mode", ["⚡ Quick Answer", "🔬 Deep Research"])

    num_questions = 5
    if mode == "🔬 Deep Research":
        num_questions = st.slider("Sub-questions to explore", 3, 8, 5)

    st.markdown("---")
    st.markdown("### 📚 About")
    st.info(
        "Quick Answer gives a direct response.\n\n"
        "Deep Research breaks your topic into sub-questions, "
        "researches each, then synthesizes a full report."
    )

# ---------- Main UI ----------
st.markdown('<p class="main-header">🔬 Deep Research Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask anything. Get a quick answer or a full research report.</p>', unsafe_allow_html=True)

query = st.text_input(
    "What do you want to know?",
    placeholder="e.g. Impact of AI on the job market",
    label_visibility="collapsed"
)
run_button = st.button("🚀 Run", type="primary", use_container_width=True)

if run_button:
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found. Please add it to your .env file.")
    elif not query.strip():
        st.warning("Please enter a question or topic first.")
    else:
        agent = DeepResearchAgent(api_key)

        if mode == "⚡ Quick Answer":
            with st.spinner("Thinking..."):
                try:
                    answer = agent.quick_answer(query)
                    st.markdown("---")
                    st.markdown('<div class="report-box">', unsafe_allow_html=True)
                    st.markdown(f"**{answer}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")

        else:
            progress = st.progress(0)
            status = st.empty()

            try:
                status.markdown("**Step 1: Generating research questions...**")
                questions = agent.generate_questions(query, num_questions)
                progress.progress(0.2)

                with st.expander("🔍 Questions being explored", expanded=True):
                    for i, q in enumerate(questions, 1):
                        st.markdown(f"{i}. {q}")

                qa_pairs = []
                for i, q in enumerate(questions, 1):
                    status.markdown(f"**Step 2: Researching {i}/{len(questions)}...**")
                    answer = agent.research_question(q)
                    qa_pairs.append({"question": q, "answer": answer})
                    progress.progress(0.2 + 0.6 * (i / len(questions)))
                    time.sleep(0.3)

                status.markdown("**Step 3: Synthesizing final report...**")
                report = agent.synthesize_report(query, qa_pairs)
                progress.progress(1.0)

                status.empty()
                progress.empty()

                st.markdown("---")
                st.markdown("### 📊 Research Report")
                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                st.markdown(report)
                st.markdown('</div>', unsafe_allow_html=True)

                report_text = f"# Research Report: {query}\n\n{report}"
                st.download_button(
                    "📥 Download Report (Markdown)",
                    data=report_text,
                    file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built with Streamlit + Groq (Llama 3.3 70B)")