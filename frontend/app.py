import streamlit as st
import requests
import json

FASTAPI_URL = "http://localhost:8000"

st.set_page_config(page_title="Sports Analytics RAG System")
st.title("Sports Analytics RAG System")
st.markdown("Ask complex queries about player performance, team statistics, and game insights.")

# Ingestion button
if st.button("Ingest Sample Documents"):
    with st.spinner("Ingesting documents..."):
        try:
            response = requests.post(f"{FASTAPI_URL}/api/ingest")
            if response.status_code == 200:
                st.success("Sample documents ingested successfully!")
            else:
                st.error(f"Failed to ingest documents: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI backend. Please ensure it is running.")

st.markdown("---")

query = st.text_input(
    "Enter your query",
    placeholder="e.g., Which team has the best defense and how does their goalkeeper compare to the league average?"
)

if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing your query..."):
            try:
                response = requests.post(
                    f"{FASTAPI_URL}/api/query",
                    json={"query": query}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success("Query processed!")

                    st.subheader("Answer")
                    st.write(data["answer"])

                    st.subheader("Details")
                    if data.get("decomposed_queries"):
                        st.write("**Decomposed Queries:**")
                        for i, q in enumerate(data["decomposed_queries"]):
                            st.write(f"{i+1}. {q}")
                    
                    st.write("**Citations:**")
                    for citation in data["citations"]:
                        st.markdown(f"**Source:** `{citation['source']}`")
                        st.markdown(f"**Snippet:** _{citation['text_snippet']}_")
                else:
                    st.error(f"Failed to get response from backend: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI backend. Please ensure it is running.")