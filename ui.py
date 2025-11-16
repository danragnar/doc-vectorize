import streamlit as st
import os
import tempfile
from sentence_transformers import SentenceTransformer
from doc_vectorize import load_vector_store, search_similar, load_document, chunk_text, generate_embeddings_local, create_vector_store, save_vector_store, extract_doc_metadata
import numpy as np
import faiss
import openai

st.title("Document Vectorizer UI")

# Ingestion section
st.header("Ingest New Documents")
uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
if st.button("Ingest Documents"):
    if uploaded_files:
        all_metadata = []
        all_embeddings = []
        
        # Create stored_documents folder
        stored_dir = 'stored_documents'
        os.makedirs(stored_dir, exist_ok=True)
        
        # If vector store exists, load existing
        if os.path.exists('vector_store.index'):
            index, existing_metadata = load_vector_store()
            all_metadata.extend(existing_metadata)
            # Reconstruct embeddings from existing metadata (approximate, since FAISS doesn't store originals)
            # For simplicity, we'll recreate from chunks, but this is inefficient
            # Better to store embeddings separately, but for now, warn
            st.warning("Appending to existing store. Note: This recreates embeddings for existing chunks.")
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            existing_texts = [item["text"] for item in existing_metadata]
            existing_embeddings = model.encode(existing_texts)
            all_embeddings.extend(existing_embeddings)
        
        import hashlib
        import shutil
        
        for uploaded_file in uploaded_files:
            file_data = uploaded_file.read()
            
            # Compute SHA256 hash
            file_hash = hashlib.sha256(file_data).hexdigest()
            ext = uploaded_file.name.split('.')[-1]
            subdir = file_hash[:2]
            subdir_path = os.path.join(stored_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            stored_filename = f"{file_hash}.{ext}"
            stored_path = os.path.join(subdir_path, stored_filename)
            
            # Save to stored_dir if not exists
            if not os.path.exists(stored_path):
                with open(stored_path, 'wb') as f:
                    f.write(file_data)
            
            # Use temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
                tmp_file.write(file_data)
                tmp_path = tmp_file.name
            
            text = load_document(tmp_path)
            doc_metadata = extract_doc_metadata(tmp_path)
            chunks = chunk_text(text)
            embeddings = generate_embeddings_local(chunks)
            
            for chunk in chunks:
                all_metadata.append({"text": chunk, "file": uploaded_file.name, "stored_path": stored_path, **doc_metadata})
            all_embeddings.extend(embeddings)
            os.unlink(tmp_path)  # Clean up
        
        all_embeddings = np.array(all_embeddings)
        index, metadata = create_vector_store(all_metadata, all_embeddings)
        save_vector_store(index, metadata)
        st.success("Ingestion complete!")
        st.rerun()  # Refresh to load new index
    else:
        st.warning("Please upload files first.")

# Check if vector store exists
if not os.path.exists('vector_store.index'):
    st.info("Upload and ingest documents to get started.")
    st.stop()

# Load the index and metadata
index, metadata = load_vector_store()

# Mode selection with tabs
tab1, tab2 = st.tabs(["Single Query", "Chat"])

with tab1:
    search_query = st.text_input("Search Query (for retrieving documents):", key="search_query")
    ai_question = st.text_input("AI Question (optional, uses search results as context):", key="ai_question")
    col1, col2 = st.columns([1,1])
    with col1:
        from_date = st.date_input("From date (optional)", key="from_date")
    with col2:
        to_date = st.date_input("To date (optional)", key="to_date")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Search Locally", key="search_local"):
            if search_query:
                model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                query_emb = model.encode([search_query])[0]
                relevant_metadata = search_similar(query_emb, index, metadata)
                
                # Filter by date
                if from_date or to_date:
                    from datetime import datetime
                    filtered = []
                    for item in relevant_metadata:
                        modified = item.get('modified')
                        if modified:
                            try:
                                mod_dt = datetime.fromisoformat(modified.split('T')[0])  # Date only
                                if from_date and mod_dt < datetime.combine(from_date, datetime.min.time()):
                                    continue
                                if to_date and mod_dt > datetime.combine(to_date, datetime.max.time()):
                                    continue
                                filtered.append(item)
                            except:
                                continue
                        else:
                            filtered.append(item)
                    relevant_metadata = filtered
                
                st.subheader("Relevant Chunks:")
                for i, item in enumerate(relevant_metadata, 1):
                    date_info = f" (modified: {item.get('modified', 'unknown')})" if item.get('modified') else ""
                    st.write(f"**{i}.** {item['text']} (from {item['file']}){date_info}")
                    # Download button
                    if 'stored_path' in item and os.path.exists(item['stored_path']):
                        with open(item['stored_path'], 'rb') as f:
                            file_data = f.read()
                        st.download_button(
                            label=f"Download {item['file']}",
                            data=file_data,
                            file_name=item['file'],
                            mime='application/octet-stream',
                            key=f"download_{i}"
                        )
                    st.divider()
            else:
                st.warning("Please enter a search query.")
    
    with col2:
        if st.button("Query AI", key="query_ai"):
            if search_query:
                try:
                    # Get context from search query
                    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                    query_emb = model.encode([search_query])[0]
                    relevant_metadata = search_similar(query_emb, index, metadata)
                    context_parts = []
                    for item in relevant_metadata:
                        file_info = f"From {item['file']}"
                        if item.get('created'):
                            file_info += f" (created: {item['created']})"
                        if item.get('modified'):
                            file_info += f" (modified: {item['modified']})"
                        context_parts.append(f"{file_info}: {item['text']}")
                    context = "\n".join(context_parts)
                    
                    # Use AI question if provided, else search query
                    question = ai_question if ai_question else search_query
                    
                    openai.api_key = os.getenv('OPENAI_API_KEY')
                    if not openai.api_key:
                        raise ValueError("OpenAI API key required")
                    
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}]
                    )
                    answer = response.choices[0].message.content
                    st.subheader("AI Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error: {e}. Make sure OPENAI_API_KEY is set.")
            else:
                st.warning("Please enter a search query.")

with tab2:
    st.header("Chat with AI")
    
    # Date filters
    col1, col2 = st.columns([1,1])
    with col1:
        chat_from_date = st.date_input("From date (optional)", key="chat_from_date")
    with col2:
        chat_to_date = st.date_input("To date (optional)", key="chat_to_date")
    
    # Inputs
    chat_search_query = st.text_input("Search Query (for retrieving documents):", key="chat_search_query")
    chat_ai_message = st.text_input("AI Message (your question to the AI):", key="chat_ai_message")
    
    # Initialize session state for history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Display chat history
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.write(f"**You:** {msg['content']}")
        else:
            st.write(f"**AI:** {msg['content']}")
    
    if st.button("Send", key="send_chat"):
        if chat_search_query and chat_ai_message:
            # Add user message to history
            st.session_state.history.append({"role": "user", "content": chat_ai_message})
            
            # Retrieve context with date filter
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            query_emb = model.encode([chat_search_query])[0]
            relevant_metadata = search_similar(query_emb, index, metadata)
            
            # Filter by date
            if chat_from_date or chat_to_date:
                from datetime import datetime
                filtered = []
                for item in relevant_metadata:
                    modified = item.get('modified')
                    if modified:
                        try:
                            mod_dt = datetime.fromisoformat(modified.split('T')[0])
                            if chat_from_date and mod_dt < datetime.combine(chat_from_date, datetime.min.time()):
                                continue
                            if chat_to_date and mod_dt > datetime.combine(chat_to_date, datetime.max.time()):
                                continue
                            filtered.append(item)
                        except:
                            continue
                    else:
                        filtered.append(item)
                relevant_metadata = filtered
            
            context_parts = []
            for item in relevant_metadata:
                file_info = f"From {item['file']}"
                if item.get('created'):
                    file_info += f" (created: {item['created']})"
                if item.get('modified'):
                    file_info += f" (modified: {item['modified']})"
                context_parts.append(f"{file_info}: {item['text']}")
            context = "\n".join(context_parts)
            
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
            st.warning("Please enter both search query and AI message.")
    
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