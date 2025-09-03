# LMVal: Multi-Metric LLM Evaluation 🧪

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Made with LangChain](https://img.shields.io/badge/Made%20with-LangChain-000000.svg)](https://www.langchain.com/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

**LMVal** is an advanced, open-source platform designed to bring rigor and clarity to the evaluation of Large Language Models (LLMs). Built with **LangChain** and **LangGraph**, it provides a comprehensive suite of metrics to quantitatively measure the quality, reliability, and accuracy of your LLM's responses, moving beyond subjective guesswork.

**Watch Demo Video:**  
See LMVal in action with this walkthrough demo:

[![Watch Demo](assets/thumbnail.png)](https://github.com/user-attachments/assets/28c3bcf8-f8f7-4d8c-a250-3bf98b33e7af)

---

## ✨ Why LMVal?

Evaluating LLM outputs is complex and often subjective. LMVal solves this by providing:

- **🔬 Multi-Metric Analysis:** Go beyond simple accuracy. Measure **Faithfulness** (hallucination detection), **Answer Relevance**, **Context Precision/Recall** (for RAG), and more.
- **📊 Actionable Insights:** Interactive visualizations, including radar charts and score distributions, help you instantly identify your model's strengths and weaknesses.
- **⚡ Concurrent Evaluation:** Leverage async processing to evaluate dozens of questions simultaneously, drastically reducing wait times.
- **🤝 Transparent Scoring:** Every score comes with a natural language explanation, so you understand the *"why"* behind the grade.
- **📁 Export Everything:** Generate detailed reports in JSON, CSV, or HTML for easy sharing and documentation.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- A [Groq](https://console.groq.com/) API key (free tier available)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nirajpatil2005/LMVal.git
    cd LMVal
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set your Groq API key:**
    ```bash
    export GROQ_API_KEY='your-api-key-here'
    # Or on Windows:
    # set GROQ_API_KEY=your-api-key-here
    ```

---

## 📊 Usage

1.  **Launch the web application:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your browser** and navigate to the local URL shown in the terminal (typically `http://localhost:8501`).

3.  **Configure your evaluation:** Select API provider, metrics, judge model, and concurrency settings.  

    <img width="1918" height="927" alt="Image" src="https://github.com/user-attachments/assets/d8cd6648-557e-47c1-8d8c-c1a5e94ddd2e" />

4.  **Upload dataset or enter manually:**  
    Provide questions, ground truths, model responses, and contexts.  

    <img width="1918" height="926" alt="Image" src="https://github.com/user-attachments/assets/6754ca06-a90e-42ca-b5ce-edff438b0946" />

5.  **Run the evaluation:** Click "Run Evaluation" and LMVal provides a detailed analysis of your LLM's performance.  

    <img width="1918" height="928" alt="Image" src="https://github.com/user-attachments/assets/274146aa-6667-4be9-ae53-e25804bae524" />
    <img width="1625" height="927" alt="Image" src="https://github.com/user-attachments/assets/c0bd234d-bada-411b-b4d1-36fa3bd60f04" />

7.  **View history:** All runs are logged with timestamp, provider, model, and scores.  

    <img width="1918" height="925" alt="Image" src="https://github.com/user-attachments/assets/510e2a0d-b38b-496b-81bb-5f6787ecb863" />

---

## 📈 Supported Evaluation Metrics

LMVal evaluates responses based on a comprehensive set of metrics:

| Metric | Description |
| :--- | :--- |
| **Accuracy** | How factually correct the response is compared to the ground truth. |
| **Faithfulness** | Measures if the response is grounded in the provided context, detecting hallucinations. |
| **Answer Relevance** | How directly and completely the response answers the original question. |
| **Context Precision** | (For RAG) How relevant the retrieved context is to the question. |
| **Context Recall** | (For RAG) How much of the necessary information was retrieved. |
| **Toxicity** | Detects harmful, offensive, or inappropriate content. |

---

## 🏗️ How It Works: Architecture

LMVal is built on a robust, agent-based architecture:

1.  **Orchestration with LangGraph:** Coordinates the multi-step evaluation process, managing the state of each question as it flows through different metric evaluations.
2.  **Evaluation Agents:** Specialized LangChain chains act as "judges" for each metric, using advanced prompting to provide consistent and reasoned scores.
3.  **Async Engine:** Handles concurrent evaluations efficiently using `asyncio`, making the process fast and scalable.
4.  **Streamlit UI:** Provides an intuitive interface for configuring runs, visualizing results, and exploring detailed feedback.

<img width="7183" height="2994" alt="Image" src="https://github.com/user-attachments/assets/a6d95ce3-e936-482b-b3c6-2ffb9d22fcfe" />

---

## 🧪 Example Evaluation

We've included a sample dataset (`demo_data.json`) to help you get started. This file contains examples of good responses, hallucinations, and factual errors, perfect for seeing LMVal in action.

```json
{
  "questions": ["What is the capital of France?"],
  "ground_truths": ["The capital of France is Paris."],
  "model_responses": ["Paris is the capital city of France."],
  "contexts": ["France, in Western Europe, has Paris as its capital..."]
}
