# 📚 Semantic Book Recommender

> 🧠 Powered by NLP, vector databases, and sentiment-aware filtering — all wrapped in an intuitive Gradio dashboard.

An LLM-powered book recommendation system that helps users discover their next favorite read by understanding **what they want to feel**, **what topics they’re interested in**, and **how they describe their ideal book**.

---
## 🚀 Project Overview

This project leverages **Large Language Models (LLMs)** to understand a user’s natural-language book query and recommend titles that match not only by content but also by **category** and **emotional tone**. It integrates semantic search, zero-shot classification, and emotion analysis to provide highly personalized recommendations.

🧩 Core techniques:
- Data preparation & feature cleaning
- Vector similarity search via embeddings
- Book category prediction via zero-shot classification
- Sentiment extraction for mood-based filtering
- Unified UI with Gradio

---
## 📊 Dataset

The dataset used in this project comes from [Kaggle - 7K Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata), which provides over 7,000 books along with metadata such as:

- Title
- Author
- Description
- Genre/Category
- Page count
- Publication date
- Ratings

This dataset served as the foundational corpus for preparing textual descriptions, performing classification, embedding generation, and running semantic similarity queries using vector databases.

> 📁 Note: Post-cleaning, approximately **5,000+ high-quality book entries** were retained for the recommender system.

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

> *"I'm looking for a book about space and time travel."*

- ✨ **Returns** science fiction titles
- ✅ Matches user-defined emotion and category through interactive filters
- 📚 Suggestions are grounded in vector similarity and LLM interpretation

---

## 📦 Folder Structure

```bash
├── data/
│   ├── cover-not-found.jpg
│   ├── tagged_desc.txt
│   ├── books_cleaned.csv
│   ├── books_with_emotions.csv
│   └── books_with_categories.csv
├── code-files/
│   ├── data-exploration.ipynb
│   ├── vector-search.ipynb
│   ├── text_classification.ipynb
│   ├── sentiment_analysis.ipynb
│   ├── gradio_dashboard.py
├── .env
├── README.md
├── requirements.txt
```

---
## 🔐 Environment Setup

### 1. Create the `.env` file:

```sh
OPENAI_API_KEY=your_key_here
```

### 2. Install dependencies:

```sh
pip install -r requirements.txt
```

### 3. Run the app:

```sh
python gradio_dashboard.py
```

---
## 🧠 Future Improvements

- Support for multi-language queries.
- Real-time usage tracking with analytics dashboard.
- Fine-tuned in-house emotion model.
- GPU-accelerated backend with async I/O.

---
## 📚 References & Learning Resources

This project was developed with insights and guidance from the following resources:

1. [FreeCodeCamp: Semantic Search + RAG + LLMs - Full Project Walkthrough](https://youtu.be/Q7mS1VHm3Yw?si=54k6R6X9sYfozqCD)
2. [Hugging Face Course: Zero-Shot Classification with Transformers](https://huggingface.co/learn/llm-course/chapter3/1?fw=pt)
3. [Dataloop Model: `emotion-english-distilroberta-base`](https://dataloop.ai/library/model/j-hartmann_emotion-english-distilroberta-base/#performance)
4. [Hugging Face Tasks: Zero-Shot Classification](https://huggingface.co/tasks/zero-shot-classification)
5. [Langchain Official Docs](https://python.langchain.com/docs/introduction/)
6. [Weaviate Docs: Understanding Vector Indexing](https://weaviate.io/developers/weaviate/concepts/vector-index)

> These resources played a key role in helping structure, build, and fine-tune various parts of the pipeline including vector search, emotion tagging, and category classification using LLMs.
