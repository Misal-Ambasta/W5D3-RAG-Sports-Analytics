from fastapi import FastAPI
from routers import rag_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Sports Analytics RAG System",
    description="A RAG system for answering complex sprts-related queries."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(rag_router.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sports Analytics API"}