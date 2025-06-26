# 📚 Semantic Book Recommender

An LLM-powered book recommendation system that helps users discover their next favorite read by understanding **what they want to feel**, **what topics they’re interested in**, and **how they describe their ideal book**.

> 🧠 Powered by NLP, vector databases, and sentiment-aware filtering — all wrapped in an intuitive Gradio dashboard.

---
## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Functional Components](#functional-components)
4. [Preparing the Text Data](#1️⃣-preparing-the-text-data)
5. [Vector Search](#2️⃣-vector-search)
6. [Text Classification (Category Prediction)](#3️⃣-text-classification-category-prediction)
7. [Sentiment Analysis](#4️⃣-sentiment-analysis)
8. [Gradio Dashboard](#5️⃣-gradio-dashboard)
9. [Example Query](#example-query)
10. [Folder Structure](#folder-structure)
11. [Environment Setup](#environment-setup)
12. [Models Used](#models-used)
13. [Future Improvements](#future-improvements)
14. [Author](#author)
15. [License](#license)
16. [Questions or Feedback](#questions-or-feedback)

---
## 🚀 Project Overview

This project leverages **Large Language Models (LLMs)** to understand a user’s natural-language book query and recommend titles that match not only by content but also by **category** and **emotional tone**. It integrates semantic search, zero-shot classification, and emotion analysis to provide highly personalized recommendations.

🧩 Core techniques:
- ✅ Data preparation & feature cleaning
- ✅ Vector similarity search via embeddings
- ✅ Book category prediction via zero-shot classification
- ✅ Sentiment extraction for mood-based filtering
- ✅ Unified UI with Gradio

---

## 🛠 Tech Stack

| Component | Tools |
|----------|-------|
| **Language** | Python |
| **IDE** | PyCharm |
| **Libraries** | `transformers`, `langchain`, `chromadb`, `pandas`, `numpy`, `gradio`, `matplotlib`, `seaborn` |
| **Models** | `facebook/bart-large-mnli`, `distilroberta-base` (emotion classifier) |
| **Dashboard** | Gradio |
| **Embeddings** | OpenAIEmbeddings |
| **Vector DB** | ChromaDB |

---

## 🔧 Functional Components

### 1️⃣ Preparing the Text Data
- Removed entries with null or meaningless descriptions.
- Normalized subtitle/title columns.
- Created a new `desc` column tagged by ISBN for unique identification.
- Retained ~5,000 clean rows post-filtering.

### 2️⃣ Vector Search
- Chunked descriptions using `CharacterTextSplitter`.
- Converted text to embeddings via `OpenAIEmbeddings`.
- Stored and queried via **ChromaDB**.
- Enabled retrieval of top-K most semantically similar books to a user’s input.

### 3️⃣ Text Classification (Category Prediction)
- Used HuggingFace `pipeline` with `facebook/bart-large-mnli`.
- Performed **zero-shot classification** to assign books into one of four simplified groups:
  - Fiction
  - Non-Fiction
  - Children’s Fiction
  - Children’s Non-Fiction
- Refilled missing categories using model predictions merged by `isbn13`.

### 4️⃣ Sentiment Analysis
- Leveraged `j-hartmann/emotion-english-distilroberta-base` model from HuggingFace.
- Extracted sentence-wise sentiment, then computed the **max score per emotion** across the entire description.
- Generated additional filtering dimensions for emotions like **joy**, **sadness**, **anger**, etc.

### 5️⃣ Gradio Dashboard
- Built an interactive frontend where users can:
  - Enter a custom book query
  - Filter by mood or category
  - Instantly get top semantic recommendations
- The dashboard integrates all stages of the backend pipeline.

---

## 🧪 Example Query

> *"I'm looking for a book about space and time travel that has a mysterious but hopeful tone."*

- ✨ **Returns** science fiction titles with aligned mood
- ✅ Matches user-defined emotion and category
- 📚 Suggestions are grounded in vector similarity and LLM interpretation

---

## 📦 Folder Structure

```bash
├── data/
│   ├── books_cleaned.csv
│   ├── books_with_emotions.csv
│   └── books_with_categories.csv
├── recommender/
│   ├── vector_search.py
│   ├── classify_books.py
│   ├── sentiment_analysis.py
│   ├── gradio_dashboard.py
│   └── utils/
├── .env
├── requirements.txt
├── README.md
└── Semantic_Book_Recommender.ipynb
