# Sports Analytics RAG System

## Problem Statement

Build a sports analytics RAG system that can answer complex queries about player performance, team statistics, and game insights from a collection of sports documents (match reports, player stats, team analyses).

## Requirements

- Implement Query Decomposition for complex multi-part questions like "Which team has the best defense and how does their goalkeeper compare to the league average?"
- Add Contextual Compression to reduce irrelevant information from retrieved documents
- Implement Basic Reranking using semantic similarity scores
- Include Citation-based responses showing which documents/sources support each claim
- Use a vector database (ChromaDB/Pinecone) for document storage
- Implement query decomposition using LLM to break complex queries into sub-questions
- Add context compression layer to filter retrieved chunks
- Simple reranking based on cosine similarity scores
- Format responses with proper citations

## Sample Queries

- "What are the top 3 teams in defense and their key defensive statistics?"
- "Compare Messi's goal-scoring rate in the last season vs previous seasons"
- "Which goalkeeper has the best save percentage in high-pressure situations?"

## Expected Deliverables

- Working RAG system with query decomposition
- Context compression implementation
- Basic reranking mechanism
- Citation formatting in responses
- Demo with sample sports data
