# personal_project_RAG_1

<img width="1072" height="95" alt="image" src="https://github.com/user-attachments/assets/97269e56-2bab-4d33-b272-bec6ee96a5ea" />
🚀 Medical RAG System (MongoDB + Sentence Transformers + DeepSeek)

A lightweight Retrieval-Augmented Generation (RAG) pipeline built on top of:

SentenceTransformer embeddings (MiniLM-L6-v2)

MongoDB Atlas Vector Search

DeepSeek Chat API

Python backend (custom pipeline)

The system allows semantic search over patient summaries and generates meaningful insights using LLM reasoning.

⭐ Features
🔹 1. End-to-End Data Pipeline

Clean medical dataset → structured summaries → embedding generation

Store vector embeddings in MongoDB Atlas

Create a vector_index for fast similarity search

🔹 2. Semantic Retrieval

Given a natural language query like:

“Who had abnormal test results for diabetes?”

The system:

Embeds the query

Runs a vector search against MongoAtlas

Returns the top-k most relevant patient summaries

🔹 3. DeepSeek LLM Integration

Retrieved chunks are passed to DeepSeek:

with safety formatting

with context window control

with optional temperature tuning

The model produces a final RAG answer.

📦 Project Structure
RAG/
│
├── embed_data.py          # create embeddings + summaries
├── insert_to_mongo.py     # upload embeddings to Atlas
├── search_mongo.py        # test semantic search pipeline
├── rag_query.py           # full RAG pipeline (search + LLM)
├── embedded_summaries.json
├── cleaned_dataset.csv
└── README.md

⚙️ Tech Stack

Python

SentenceTransformer

MongoDB Atlas Vector Search

DeepSeek Chat API

Requests

Uvicorn (optional server)

🧠 How It Works

Dataset Cleaning

Summaries Generated for Each Row

Embeddings (384-dim) computed using MiniLM

Stored in MongoDB

Vector index created ($vectorSearch)

Queries embedded → semantic search → top-K results

DeepSeek produces final answer

📝 Example Query
Ask something: Who had abnormal test results for diabetes?


Output:

Top matches:
1. Patient Christopher Velasquez...
2. Patient Monica Collins...
3. Patient Virginia Mercado...

❤️ Status

This project serves as the foundation for the more powerful Agentic RAG System.

