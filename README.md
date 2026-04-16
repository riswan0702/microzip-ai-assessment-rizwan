# microzip-ai-assessment-rizwan

> **MicroZip IT Solutions — AI/ML Engineer Technical Assessment**  
> Candidate: Rizwan P P

---

##  Live Demo & Walkthrough

| Component | URL |
|-----------|-----|
| **Live Demo (Streamlit)** | https://huggingface.co/spaces/rizwan0702/microzip-ai-assessment-rizwan |
| **Loom Walkthrough** | https://www.loom.com/share/7ae698852aa9461094e227b609e5444b |
| **GitHub Repo** | https://github.com/riswan0702/microzip-ai-assessment-rizwan |

---

## Project Structure

```
microzip-ai-assessment-rizwan/
│
├── part1_ml_model.ipynb        # Part 1: ML training, EDA, evaluation
├── best_model.joblib           # Saved best model
├── scaler.joblib               # Feature scaler
├── feature_names.joblib        # Feature names
│
├── app.py                      # Part 2: RAG AI Agent (Streamlit)
│
├── sample_document.txt         # Sample document for testing
├── requirements.txt
└── README.md
```

---

## ⚙️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/riswan0702/microzip-ai-assessment-rizwan
cd microzip-ai-assessment-rizwan
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

Open → `http://localhost:8501`

---

## How to Use the RAG Agent

1. Enter your **Groq API key** in the sidebar (free at [console.groq.com](https://console.groq.com))
2. Upload any **PDF or TXT** document
3. Click **Build Knowledge Base**
4. Ask questions about the document
5. Ask follow-up questions — the agent remembers the conversation!

---

## Part 1 — ML Model

Open and run the notebook:
```bash
jupyter notebook part1_ml_model.ipynb
```

**What it covers:**
- Exploratory Data Analysis on Heart Disease dataset
- Data preprocessing: imputation, encoding, scaling
- Training 3 models: Logistic Regression, Random Forest, XGBoost
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Best model saved as `best_model.joblib`

**Best Model: XGBoost** — highest F1 Score and ROC-AUC.

---

## RAG Agent Architecture

```
User Question
      │
      ▼
Streamlit UI
      │
      ▼
FAISS Vector Store  ◄── HuggingFace MiniLM Embeddings
      │
      ▼
Top-K Relevant Chunks 
      │
      ▼
Groq LLaMA 3.3 70B 
      │
      ▼
Answer + Source Context + Conversation Memory
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn, XGBoost |
| Embeddings | sentence-transformers (MiniLM-L6-v2) |
| Vector DB | FAISS |
| LLM | Groq LLaMA 3.3 70B (free) |
| Orchestration | LangChain |
| Web UI | Streamlit |


