import streamlit as st
import os
import tempfile
from sentence_transformers import SentenceTransformer
from doc_vectorize import load_vector_store, search_similar, query_pipeline, load_document, chunk_text, generate_embeddings_local, create_vector_store, save_vector_store
import numpy as np
import faiss

st.title("Document Vectorizer UI")

# Ingestion section
st.header("Ingest New Documents")
uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
if st.button("Ingest Documents"):
    if uploaded_files:
        all_chunks = []
        all_embeddings = []
        
        # If vector store exists, load existing
        if os.path.exists('vector_store.index'):
            index, existing_chunks = load_vector_store()
            all_chunks.extend(existing_chunks)
            # Reconstruct embeddings from existing index (approximate, since FAISS doesn't store originals)
            # For simplicity, we'll recreate from chunks, but this is inefficient
            # Better to store embeddings separately, but for now, warn
            st.warning("Appending to existing store. Note: This recreates embeddings for existing chunks.")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            existing_embeddings = model.encode(existing_chunks)
            all_embeddings.extend(existing_embeddings)
        
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            text = load_document(tmp_path)
            chunks = chunk_text(text)
            embeddings = generate_embeddings_local(chunks)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
            os.unlink(tmp_path)  # Clean up
        
        all_embeddings = np.array(all_embeddings)
        index, chunks = create_vector_store(all_chunks, all_embeddings)
        save_vector_store(index, chunks)
        st.success("Ingestion complete!")
        st.rerun()  # Refresh to load new index
    else:
        st.warning("Please upload files first.")

# Check if vector store exists
if not os.path.exists('vector_store.index'):
    st.info("Upload and ingest documents to get started.")
    st.stop()

# Load the index and chunks
index, chunks = load_vector_store()

query = st.text_input("Enter your query:")

col1, col2 = st.columns(2)

with col1:
    if st.button("Search Locally"):
        if query:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_emb = model.encode([query])[0]
            relevant_chunks = search_similar(query_emb, index, chunks)
            st.subheader("Relevant Chunks:")
            for i, chunk in enumerate(relevant_chunks, 1):
                st.write(f"**{i}.** {chunk}")
                st.divider()
        else:
            st.warning("Please enter a query.")

with col2:
    if st.button("Query with AI"):
        if query:
            try:
                answer = query_pipeline(query, index, chunks)
                st.subheader("AI Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}. Make sure OPENAI_API_KEY is set.")
        else:
            st.warning("Please enter a query.")