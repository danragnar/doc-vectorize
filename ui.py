import streamlit as st
import os
import tempfile
from sentence_transformers import SentenceTransformer
from doc_vectorize import load_vector_store, search_similar, load_document, chunk_text, generate_embeddings_local, create_vector_store, save_vector_store
import numpy as np
import faiss
import openai

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
            # Reconstruct embeddings from existing chunks (approximate, since FAISS doesn't store originals)
            # For simplicity, we'll recreate from chunks, but this is inefficient
            # Better to store embeddings separately, but for now, warn
            st.warning("Appending to existing store. Note: This recreates embeddings for existing chunks.")
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
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

# Mode selection with tabs
tab1, tab2 = st.tabs(["Single Query", "Chat"])

with tab1:
    query = st.text_input("Enter your query:", key="single_query")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Search Locally", key="search_local"):
            if query:
                model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                query_emb = model.encode([query])[0]
                relevant_chunks = search_similar(query_emb, index, chunks)
                st.subheader("Relevant Chunks:")
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.write(f"**{i}.** {chunk}")
                    st.divider()
            else:
                st.warning("Please enter a query.")
    
    with col2:
        if st.button("Query AI (Single)", key="query_ai"):
            if query:
                try:
                    # Simulate query_pipeline
                    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                    query_emb = model.encode([query])[0]
                    relevant_chunks = search_similar(query_emb, index, chunks)
                    context = "\n".join(relevant_chunks)
                    
                    openai.api_key = os.getenv('OPENAI_API_KEY')
                    if not openai.api_key:
                        raise ValueError("OpenAI API key required")
                    
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}]
                    )
                    answer = response.choices[0].message.content
                    st.subheader("AI Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error: {e}. Make sure OPENAI_API_KEY is set.")
            else:
                st.warning("Please enter a query.")

with tab2:
    st.header("Chat with AI")
    
    # Initialize session state for history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Display chat history
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.write(f"**You:** {msg['content']}")
        else:
            st.write(f"**AI:** {msg['content']}")
    
    # Input for new message
    new_message = st.text_input("Your message:", key="chat_input")
    
    if st.button("Send", key="send_chat"):
        if new_message:
            # Add user message to history
            st.session_state.history.append({"role": "user", "content": new_message})
            
            # Retrieve context
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            query_emb = model.encode([new_message])[0]
            relevant_chunks = search_similar(query_emb, index, chunks)
            context = "\n".join(relevant_chunks)
            
            # Build messages
            messages = [{"role": "system", "content": f"Context from documents: {context}"}] + st.session_state.history
            
            # Call OpenAI
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if not openai.api_key:
                st.error("OPENAI_API_KEY not set.")
            else:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages
                    )
                    answer = response.choices[0].message.content
                    st.session_state.history.append({"role": "assistant", "content": answer})
                    st.rerun()  # Refresh to show new message
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a message.")
    
    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.history = []
        st.rerun()

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