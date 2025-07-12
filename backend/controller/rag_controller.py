import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Core Langchain components
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Community and Integrations
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# components for RAG
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
print(os.getenv("GEMINI_API_KEY"))
google_api_key = os.getenv("GEMINI_API_KEY")
if not google_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

from langchain_google_genai import ChatGoogleGenerativeAI


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
vector_db_path = "./data/chroma_db"
os.makedirs(vector_db_path,exist_ok=True)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    max_retries=2,
    google_api_key=google_api_key
)

class RAGService:
    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store = self.__init__vector_store()
        self.retriever = self._get_compression_retriever()
        self.qa_chain = self._get_rag_chain()

    def __init__vector_store(self):
        #Initialize or load ChromaDB
        if os.path.exists(os.path.join(vector_db_path, "index")):
            print("Loading existing ChromaDB...")
            return Chroma(
                persist_directory=vector_db_path,
                embedding_function=self.embedding
            )
        else:
            print("Initializing new ChromaDB (empty for now)...")
            return Chroma(
                embedding_function=self.embedding,
                persist_directory=vector_db_path
            )


    def ingest_documents(self, doc_text: List[str], metadata: List[Dict[str, Any]]):
        # Create Document objects
        documents = []
        for text, meta in zip(doc_text, metadata):
            documents.append(Document(page_content=text, metadata=meta))

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        print(f"Ingesting {len(chunks)} chunks into ChromaDB...")

        # Add chunks to vector store
        self.vector_store.add_documents(chunks)
        self.vector_store.persist() #persist the changes
        print("Documents ingested successfully.")

    def _get_base_retriever(self):
        #set up a basic retriever froom the vector store
        return self.vector_store.as_retriever(search_kwargs={"k":5})

    def _get_compression_retriever(self):
        # Contextual Compression using LLMChainExtractor
        compressor = LLMChainExtractor.from_llm(llm)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self._get_base_retriever()
        )

    def _get_query_decomposition_chain(self):
        # Query Decomposition Chain using LCEL
        query_decomposition_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a sports analytics assistant. Break down the user's complex query into a set of simpler, atomic queries.
            Return a comma-separated list of the sub-queries.
            Example: 'What are the top 3 teams in defense and their key defensive statistics?' -> 'top 3 defensive teams, key defensive statistics for these teams'
            Query: {query}"""),
            ("human", "{query}")
        ])
        return query_decomposition_prompt | llm | StrOutputParser()

    def _get_rag_chain(self):
        # Answer Generation Chain
        qa_system_prompt = """
        You are a helpful sports analytics assistant. Use the following retrieved context to answer the user's question.
        Each document chunk in the context is from a specific source, indicated by 'Source: [document_id]'.
        
        When providing the answer, you MUST cite the source of any information you use. For each piece of information,
        add a citation like so: [Source: document_id].
        
        Example: "The team's win percentage is 85% [Source: report_A]. They also have a goal difference of +25 [Source: report_B]."
        
        If the context does not contain the information needed, state that you cannot answer the question based on the provided documents.
        
        Context:
        {context}

        Question:
        {input}
        """
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            ("human", "{input}")
        ])

        # Create a simple chain to combine documents and generate answer
        document_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Create the full RAG Chain
        # This chain will take an input, retrieve documents and then pass them to  the document_chain
        rag_chain = create_retrieval_chain(self.retriever, document_chain)

        return rag_chain

    def process_query(self, query:str) -> Dict[str, Any]:
        # 1. Query Decomposition
        decomposer_chain = self._get_query_decomposition_chain()
        decomposed_queries_str = decomposer_chain.invoke({"query": query})
        decomposed_queries = [q.strip() for q in decomposed_queries_str.split(',')]

        # 2. RAG Chain Execution
        # The create_retrieval_chain directly handles retrieval and document stuffing
        # We pass the original query to the RAG chain.
        # The retriever within the RAG chain (self.retriever) will apply compression.
        
        # The 'input' key matches what create_retrieval_chain expects
        result = self.qa_chain.invoke({"input": query})
        
        # The 'result' dictionary from create_retrieval_chain contains 'answer' and 'context' (retrieved documents)
        answer = result["answer"]
        retrieved_docs = result["context"]

        # 3. Extract Citations
        citations = []
        for doc in retrieved_docs:
            source_id = doc.metadata.get('source')
            if source_id and f"[Source: {source_id}]" in answer: # Check if the source was actually used in the answer
                citations.append({
                    "source": source_id,
                    "text_snippet": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                })
        
        return {
            "answer": answer,
            "citations": citations,
            "decomposed_queries": decomposed_queries
        }
