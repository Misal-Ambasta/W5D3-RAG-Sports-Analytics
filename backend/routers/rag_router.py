from fastapi import APIRouter
from controller.rag_controller import RAGService
from models.schemas import QueryRequest, RAGResponse

router = APIRouter()
rag_service = RAGService()

@router.post("/ingest")
def ingest_documents_endpoint():
    # Placeholder for ingesting documents
    # Example data:
    docs = [
        "Team A has the best defense, allowing only 1.2 goals per game. Their goalkeeper, John Doe, has a save percentage of 85% [Source: report_A].",
        "Team B's key defensive statistics include a high number of tackles and interceptions, leading the league in both. Their goalkeeper, Jane Smith, has a save percentage of 70% [Source: report_B].",
        "Lionel Messi scored 50 goals in the 2022-2023 season and 35 goals in the 2021-2022 season. His goal-scoring rate was significantly higher in the last season [Source: player_stats_messi].",
        "Goalkeeper Mark Johnson has the best save percentage in high-pressure situations, with a 92% success rate on penalty kicks [Source: goalkeeper_report_2023]."
    ]
    metadata = [
        {"source": "report_A"},
        {"source": "report_B"},
        {"source": "player_stats_messi"},
        {"source": "goalkeeper_report_2023"}
    ]
    rag_service.ingest_documents(docs, metadata)
    return {"message": "Documents ingested successfully."}


@router.post("/query", response_model=RAGResponse)
async def query_rag_system(request: QueryRequest):
    result = rag_service.process_query(request.query)
    return RAGResponse(**result)