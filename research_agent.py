"""
Deep Research Agent
--------------------
Uses Groq's free API (Llama 3.3 70B) to research any topic:
1. Breaks the topic into sub-questions
2. Researches each sub-question
3. Synthesizes everything into one final report

Setup:
1. pip install -r requirements.txt
2. Create a .env file with: GROQ_API_KEY=your_key_here
3. Run: python research_agent.py
"""

import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"


class DeepResearchAgent:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.reports_dir = "research_reports"
        os.makedirs(self.reports_dir, exist_ok=True)

    def _ask(self, prompt: str, max_tokens: int = 1024) -> str:
        """Send a single prompt to the model and return the text response."""
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def generate_questions(self, topic: str, num_questions: int = 5) -> list:
        """Step 1: Break the topic into focused sub-questions."""
        prompt = f"""You are a research planner. Break down this topic into
{num_questions} focused, non-overlapping research questions.

Topic: {topic}

Return ONLY the questions, one per line, no numbering, no extra text."""

        result = self._ask(prompt, max_tokens=400)
        questions = [q.strip("- ").strip() for q in result.split("\n") if q.strip()]
        return questions[:num_questions]

    def research_question(self, question: str) -> str:
        """Step 2: Research a single sub-question in depth."""
        prompt = f"""Answer this research question thoroughly and factually,
in 3-5 sentences. Be specific and avoid filler.

Question: {question}"""

        return self._ask(prompt, max_tokens=500)

    def synthesize_report(self, topic: str, qa_pairs: list) -> str:
        """Step 3: Combine all findings into one structured report."""
        context = "\n\n".join(
            f"Q: {qa['question']}\nA: {qa['answer']}" for qa in qa_pairs
        )

        prompt = f"""Using the research findings below, write a clear,
well-structured report on the topic. Include a short summary, key findings,
and a brief conclusion. Do not repeat the raw Q&A format - synthesize it
into flowing sections.

Topic: {topic}

Research Findings:
{context}

Write the final report:"""

        return self._ask(prompt, max_tokens=1500)

    def conduct_deep_research(self, topic: str, num_questions: int = 5) -> dict:
        print(f"\n{'='*70}")
        print(f"🔬 Deep Research: {topic}")
        print(f"{'='*70}\n")

        print("Step 1: Generating research questions...")
        questions = self.generate_questions(topic, num_questions)
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")

        print("\nStep 2: Researching each question...")
        qa_pairs = []
        for i, q in enumerate(questions, 1):
            print(f"  Researching {i}/{len(questions)}...")
            answer = self.research_question(q)
            qa_pairs.append({"question": q, "answer": answer})
            time.sleep(1)  # gentle rate limiting

        print("\nStep 3: Synthesizing final report...")
        report = self.synthesize_report(topic, qa_pairs)

        result = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "questions_explored": questions,
            "qa_pairs": qa_pairs,
            "report": report,
        }

        self._save_json(result)
        self._save_markdown(result)

        print(f"\n{'='*70}")
        print("✅ Research complete!")
        print(f"{'='*70}\n")

        return result

    def quick_answer(self, question: str) -> str:
        """Direct mode: skip the multi-step research, just answer directly."""
        prompt = f"""Answer this question directly and concisely,
in 2-3 sentences maximum. No preamble.

Question: {question}"""
        return self._ask(prompt, max_tokens=200)

    def _save_json(self, result: dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.reports_dir, f"{timestamp}_research.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved: {path}")

    def _save_markdown(self, result: dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.reports_dir, f"{timestamp}_report.md")

        lines = [
            f"# Research Report: {result['topic']}",
            f"\n*Generated: {result['timestamp']}*",
            f"*Model: {result['model']}*\n",
            "---\n",
            result["report"],
            "\n---\n## Questions Explored\n",
        ]
        for i, q in enumerate(result["questions_explored"], 1):
            lines.append(f"{i}. {q}")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"💾 Saved: {path}")


def main():
    if not GROQ_API_KEY:
        print("\n❌ ERROR: GROQ_API_KEY not found!")
        print("1. Create a .env file in this folder")
        print("2. Add this line: GROQ_API_KEY=your_key_here")
        print("3. Get a free key at: https://console.groq.com/\n")
        return

    agent = DeepResearchAgent(GROQ_API_KEY)

    print("\n" + "=" * 70)
    print("🤖 Deep Research Agent")
    print("=" * 70)
    print("Commands:")
    print("  research <topic>   - Full multi-step deep research")
    print("  <any question>     - Quick direct answer")
    print("  exit / quit        - Stop")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("💬 You: ").strip()

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("👋 Goodbye!")
                break

            if user_input.lower().startswith("research "):
                topic = user_input[9:].strip()
                result = agent.conduct_deep_research(topic)
                print("\n" + result["report"] + "\n")
            else:
                answer = agent.quick_answer(user_input)
                print(f"\n💡 {answer}\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()