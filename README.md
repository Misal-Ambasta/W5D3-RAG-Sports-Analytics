# Sports Analytics RAG System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for sports analytics, capable of answering complex queries about player performance, team statistics, and game insights. The backend is built with FastAPI following the MVC pattern, using LangChain v0.3+ for RAG functionality, ChromaDB for vector storage, and a Streamlit frontend for user interaction.

## Features

- **Query Decomposition**: Breaks down complex queries into sub-questions for precise answers
- **Contextual Compression**: Filters irrelevant information from retrieved documents using LLMChainExtractor
- **Vector Database**: Uses ChromaDB for efficient document storage and retrieval
- **Citation-based Responses**: Includes source citations for transparency
- **Modern Tech Stack**: FastAPI backend with Streamlit frontend
- **Google Gemini Integration**: Uses Gemini 2.0 Flash for LLM capabilities

## Sample Queries
- "What are the top 3 teams in defense and their key defensive statistics?"
- "Compare Messi's goal-scoring rate in the last season vs previous seasons"
- "Which goalkeeper has the best save percentage in high-pressure situations?"

## Project Structure
```
W5D3-RAG-Sports-Analytics/
├── backend/
│   ├── controller/
│   │   ├── __init__.py
│   │   └── rag_controller.py      # Core RAG logic and service
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py             # Pydantic models for API
│   ├── routers/
│   │   ├── __init__.py
│   │   └── rag_router.py          # FastAPI routes
│   ├── data/
│   │   └── chroma_db/             # ChromaDB storage
│   └── main.py                    # FastAPI application entry point
├── frontend/
│   └── app.py                     # Streamlit frontend
├── data/
│   └── chroma_db/                 # Vector database storage
├── requirements.txt               # Project dependencies
├── problem_statement.md           # Project requirements
└── README.md                      # This file
```

## Prerequisites

- Python 3.9+
- Google Gemini API Key
- pip for installing dependencies

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd W5D3-RAG-Sports-Analytics
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

## Dependencies
- fastapi - Web framework for building APIs
- uvicorn - ASGI server for FastAPI
- streamlit - Frontend web app framework
- langchain>=0.3 - LLM application framework
- langchain-core - Core LangChain components
- langchain-community - Community integrations
- langchain-huggingface - HuggingFace embeddings
- langchain-google-genai - Google Gemini integration
- sentence-transformers - Embedding models
- pydantic>=2.0 - Data validation
- chromadb - Vector database
- python-dotenv - Environment variable management

## Running the Application

### Backend (FastAPI)
```bash
cd backend
uvicorn main:app --reload
```
The backend will run on `http://localhost:8000`

### Frontend (Streamlit)
```bash
cd frontend
streamlit run app.py
```
The frontend will run on `http://localhost:8501`

## API Endpoints

### 1. Root Endpoint
- **GET** `/`
- Returns welcome message

### 2. Ingest Documents
- **POST** `/api/ingest`
- Ingests sample sports documents into the vector database
- No request body required

### 3. Query RAG System
- **POST** `/api/query`
- Processes user queries using RAG
- **Request Body:**
  ```json
  {
    "query": "Your sports analytics question here"
  }
  ```
- **Response:**
  ```json
  {
    "answer": "Generated answer with citations",
    "citations": [
      {
        "source": "document_id",
        "text_snippet": "Relevant text snippet..."
      }
    ],
    "decomposed_queries": ["sub-query 1", "sub-query 2"]
  }
  ```

## Usage

1. **Start both backend and frontend** as described above
2. **Open the Streamlit interface** in your browser (`http://localhost:8501`)
3. **Ingest Sample Data:**
   - Click "Ingest Sample Documents" to load predefined sports data
4. **Ask Questions:**
   - Enter queries in the text input field
   - Click "Get Answer" to receive responses with citations

## System Architecture

The system implements a sophisticated RAG pipeline with:

1. **Document Ingestion**: Text splitting and vector embedding using HuggingFace models
2. **Query Processing**: Multi-step process including decomposition and retrieval
3. **Contextual Compression**: LLM-based filtering of retrieved documents
4. **Answer Generation**: Google Gemini-powered response generation with citations

## Sample Data

The system includes predefined sample documents covering:
- Team defensive statistics
- Player performance data (including Messi's goal statistics)
- Goalkeeper performance metrics
- League statistics

## Configuration

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: Google Gemini 2.0 Flash
- **Vector Database**: ChromaDB with persistent storage
- **Chunk Size**: 1000 characters with 100 character overlap
- **Retrieval**: Top 5 documents with contextual compression

## Development Notes

- Uses LangChain v0.3+ for latest API compatibility
- ChromaDB provides persistent vector storage
- Contextual compression reduces noise in retrieved documents
- Citation tracking ensures transparency in responses
- MVC pattern separates concerns for maintainability

## Troubleshooting

1. **Connection Error**: Ensure both backend and frontend are running
2. **API Key Error**: Verify GEMINI_API_KEY is set in environment variables
3. **Import Errors**: Ensure all dependencies are installed with correct versions
4. **Database Issues**: Check ChromaDB permissions and storage directory

## Extending the System

- **Enhanced Query Decomposition**: Add more sophisticated query parsing
- **Advanced Reranking**: Implement cross-encoder models
- **Additional Data Sources**: Support for CSV, JSON, and database connections
- **Authentication**: Add user management and API key authentication
- **Monitoring**: Add logging and performance metrics
