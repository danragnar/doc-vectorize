import os
import sys
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA
import faiss

def load_pdf(file_path):
    """Extract text from PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_docx(file_path):
    """Extract text from DOCX file."""
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                text += cell.text + "\n"
    return text

def load_document(file_path):
    """Load document based on file extension."""
    if file_path.endswith('.pdf'):
        return load_pdf(file_path)
    elif file_path.endswith('.docx'):
        return load_docx(file_path)
    else:
        raise ValueError("Unsupported file type")

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def generate_embeddings_local(chunks, model_name='all-MiniLM-L6-v2'):
    """Generate embeddings using local sentence-transformers model."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings

def create_vector_store(chunks, embeddings):
    """Create FAISS vector store."""
    if embeddings.size == 0:
        raise ValueError("No embeddings to create vector store")
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    # Store chunks as metadata
    return index, chunks

def save_vector_store(index, chunks, save_path='vector_store.index'):
    """Save FAISS index and chunks."""
    faiss.write_index(index, save_path)
    with open(save_path + '.chunks', 'w') as f:
        for chunk in chunks:
            f.write(chunk + '\n---\n')

def load_vector_store(load_path='vector_store.index'):
    """Load FAISS index and chunks."""
    index = faiss.read_index(load_path)
    with open(load_path + '.chunks', 'r') as f:
        content = f.read()
        chunks = content.split('\n---\n')[:-1]  # Remove last empty
    return index, chunks

def search_similar(query_embedding, index, chunks, k=5):
    """Search for similar chunks."""
    query_emb = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_emb, k)
    results = [chunks[i] for i in indices[0]]
    return results

def query_pipeline(query, index, chunks, model_name='all-MiniLM-L6-v2', openai_api_key=None):
    """Query pipeline with retrieval and ChatGPT."""
    import openai

    # Generate query embedding
    model = SentenceTransformer(model_name)
    query_emb = model.encode([query])[0]

    # Retrieve similar chunks
    relevant_chunks = search_similar(query_emb, index, chunks)
    context = "\n".join(relevant_chunks)

    # Use ChatGPT for answer
    if not openai_api_key:
        openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OpenAI API key required")

    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}]
    )
    return response.choices[0].message.content

def main():
    if len(sys.argv) < 2:
        print("Usage: python doc_vectorize.py <command> [args]")
        print("Commands: ingest <input_dir>, query <question>, search <question>")
        return
    
    command = sys.argv[1]
    
    if command == 'ingest':
        input_dir = sys.argv[2]
        all_chunks = []
        all_embeddings = []
        
        for file in os.listdir(input_dir):
            if file.endswith(('.pdf', '.docx')):
                file_path = os.path.join(input_dir, file)
                text = load_document(file_path)
                chunks = chunk_text(text)
                embeddings = generate_embeddings_local(chunks)
                all_chunks.extend(chunks)
                all_embeddings.extend(embeddings)
        
        # Convert to numpy array
        import numpy as np
        all_embeddings = np.array(all_embeddings)
        index, chunks = create_vector_store(all_chunks, all_embeddings)
        save_vector_store(index, chunks)
        print("Ingestion complete")
    
    elif command == 'query':
        question = ' '.join(sys.argv[2:])
        index, chunks = load_vector_store()
        answer = query_pipeline(question, index, chunks)
        print(answer)
    
    elif command == 'search':
        question = ' '.join(sys.argv[2:])
        index, chunks = load_vector_store()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_emb = model.encode([question])[0]
        relevant_chunks = search_similar(query_emb, index, chunks)
        print("Relevant chunks:")
        for i, chunk in enumerate(relevant_chunks, 1):
            print(f"{i}. {chunk}\n---")

if __name__ == "__main__":
    main()