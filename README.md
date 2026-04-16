# microzip-ai-assessment-rizwan

> **MicroZip IT Solutions — AI/ML Engineer Technical Assessment**  
> Candidate: Rizwan P P  
> Submission Date: *(fill in)*

---

## 🗂️ Project Structure

```
microzip-ai-assessment-rizwan/
│
├── part1_ml_model.ipynb        # ML training, EDA, evaluation
├── best_model.joblib           # Saved best model (generated after running notebook)
├── scaler.joblib               # StandardScaler (generated after running notebook)
├── feature_names.joblib        # Feature names for inference
│
├── part2_rag_agent/
│   ├── app.py                  # Streamlit RAG chatbot UI
│   └── api.py                  # FastAPI REST API (bonus)
│
├── sample_document.pdf         # Sample PDF used for testing the agent
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Live Deployment

| Component | URL |
|-----------|-----|
| **RAG Agent (Streamlit)** | *(your deployed URL here — e.g. https://huggingface.co/spaces/your-name/microzip-rag)* |
| **FastAPI Docs** | *(your deployed URL + /docs)* |

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.10+
- A free Groq API key from [console.groq.com](https://console.groq.com)

### 1. Clone the repo
```bash
git clone https://github.com/your-username/microzip-ai-assessment-rizwan
cd microzip-ai-assessment-rizwan
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variable (optional)
```bash
export GROQ_API_KEY=gsk_your_key_here
```
Or enter it in the Streamlit UI sidebar.

---

## 🧪 Part 1 — ML Model

### Run the Notebook
```bash
jupyter notebook part1_ml_model.ipynb
```

**What it does:**
- Loads the Heart Disease dataset (UCI ML Repository)
- Performs full EDA: distributions, null analysis, correlation heatmap
- Preprocesses: imputation, one-hot encoding, scaling
- Trains 3 models: Logistic Regression, Random Forest, XGBoost
- Evaluates: Accuracy, Precision, Recall, F1, ROC-AUC + Confusion Matrix
- Selects best model (XGBoost) and saves it as `best_model.joblib`

---

## 🤖 Part 2 — RAG AI Agent

### Run Streamlit App
```bash
streamlit run part2_rag_agent/app.py
```
Open → `http://localhost:8501`

**Steps:**
1. Enter your Groq API key in the sidebar
2. Upload any PDF or TXT document
3. Click "Build Knowledge Base"
4. Ask questions — including follow-ups!

### Run FastAPI (Bonus)
```bash
uvicorn part2_rag_agent.api:app --reload --port 8000
```
Open API docs → `http://localhost:8000/docs`

**API Usage:**
```bash
# 1. Upload a document
curl -X POST http://localhost:8000/upload \
  -F "file=@sample_document.pdf"

# 2. Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "groq_api_key": "gsk_..."}'

# 3. Ask a follow-up (memory is maintained)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can you give more detail on that?"}'

# 4. Get history
curl http://localhost:8000/history
```

---

## 🐳 Docker

```bash
# Build
docker build -t microzip-rag .

# Run Streamlit
docker run -p 8501:8501 -e GROQ_API_KEY=gsk_... microzip-rag

# Run FastAPI instead
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... microzip-rag \
  uvicorn part2_rag_agent.api:app --host 0.0.0.0 --port 8000
```

---

## ☁️ Deployment (Hugging Face Spaces)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → New Space
2. Choose **Streamlit** as SDK
3. Upload all files or connect your GitHub repo
4. Add secret: `GROQ_API_KEY` in Space Settings → Repository Secrets
5. The app auto-deploys — copy the live URL

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Streamlit UI / FastAPI
    │
    ▼
LangChain ConversationalRetrievalChain
    │         │
    │    ConversationBufferWindowMemory (last 5 turns)
    │
    ▼
FAISS Vector Store  ◄── HuggingFace MiniLM Embeddings
    │
    ▼
Top-K Relevant Chunks (MMR search)
    │
    ▼
Groq LLaMA3-8B  ──►  Answer + Source Context
```

---

## 📊 Model Performance Summary (Part 1)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| XGBoost | ~0.87 | ~0.88 | ~0.87 | ~0.87 | ~0.93 |
| Random Forest | ~0.85 | ~0.86 | ~0.85 | ~0.85 | ~0.91 |
| Logistic Regression | ~0.82 | ~0.83 | ~0.82 | ~0.82 | ~0.89 |

*Exact values may vary slightly per run due to random seeds.*

**Best Model: XGBoost** — highest F1 and ROC-AUC, best at balancing precision and recall in medical classification.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | scikit-learn, XGBoost |
| EDA/Viz | pandas, matplotlib, seaborn |
| Embeddings | sentence-transformers (MiniLM-L6-v2) |
| Vector DB | FAISS |
| LLM | Groq LLaMA3-8B (free tier) |
| Orchestration | LangChain |
| Web UI | Streamlit |
| REST API | FastAPI |
| Deployment | Hugging Face Spaces / Render |

---

## 📬 Contact

**Candidate:** Rizwan P P  
**Submission to:** info@microzipitsolutions.com  
**Subject:** AI/ML Assessment Submission - Rizwan P P
