from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query about sports analytics.")

class DocumentChunk(BaseModel):
    text: str
    source: str
    
class Citation(BaseModel):
    source: str
    text_snippet: str
    
class RAGResponse(BaseModel):
    answer: str
    citations: List[Citation]
    decomposed_queries: Optional[List[str]] = None