# Document Vectorizer

A tool for ingesting documents (PDF/DOCX) into a vector store and querying them using semantic search, with optional AI-powered answers via OpenAI.

## Features

- **Ingestion**: Extract text from PDF and DOCX files, chunk it, generate embeddings, and store in a FAISS vector index.
- **Search**: Perform semantic search to retrieve relevant document chunks locally.
- **Query**: Use retrieved chunks as context for AI-generated answers (requires OpenAI API key).
- **UI**: Web-based interface for easy interaction (upload documents, search, query).

## Installation

1. Clone or download the repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. (Optional) Set your OpenAI API key for AI queries:
   ```
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### CLI

#### Ingest Documents
Process all PDF/DOCX files in a directory:
```
python doc_vectorize.py ingest /path/to/document/directory
```

#### Search Locally
Retrieve relevant chunks for a query without AI:
```
python doc_vectorize.py search "your query here"
```

#### Query with AI
Get an AI-generated answer using retrieved context:
```
python doc_vectorize.py query "your question here"
```
*Requires OPENAI_API_KEY.*

#### Chat with AI
Start an interactive chat session with conversation history:
```
python doc_vectorize.py chat
```
Type your questions, and the AI will respond with context from documents. Type 'exit' or 'quit' to end. *Requires OPENAI_API_KEY.*

### Web UI

Launch the interactive web interface:
```
streamlit run ui.py
```
- Upload and ingest new documents.
- Enter queries to search locally or query with AI.
- Access at http://localhost:8501 (default).

## Requirements

- Python 3.8+
- OpenAI API key (for `query` command and UI AI feature)
- Dependencies listed in `requirements.txt`

## Examples

1. Ingest documents:
   ```
   python doc_vectorize.py ingest ./documents/
   ```

2. Search for "machine learning":
   ```
   python doc_vectorize.py search "machine learning"
   ```

3. Ask a question:
   ```
   python doc_vectorize.py query "What is the capital of France?"
   ```

4. Start a chat session:
   ```
   python doc_vectorize.py chat
   ```

5. Launch UI:
   ```
   streamlit run ui.py
   ```

## Notes

- The vector store is saved as `vector_store.index` and `vector_store.index.chunks`.
- Ingestion supports PDF and DOCX files.
- Embeddings are generated using the `paraphrase-multilingual-MiniLM-L12-v2` model, which supports multiple languages including Swedish.
- For large document sets, ingestion may take time due to embedding generation.
- The UI allows incremental ingestion by uploading files.