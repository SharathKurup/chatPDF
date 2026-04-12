# chatPDF

A conversational AI web application built with Streamlit that allows you to chat with PDF documents using local language models, semantic search, and intelligent result re-ranking.

## Features

- 🌐 **Web Interface**: User-friendly Streamlit web app for easy PDF upload and chat
- 📄 **PDF Processing**: Extract and process text from PDF documents
- 🔍 **Semantic Search with FAISS**: Fast vector similarity search using FAISS for efficient retrieval
- 🎯 **Intelligent Re-ranking**: Uses FlashRank to re-rank search results for better relevance
- 💬 **Conversational Interface**: Ask questions about your PDF in a natural, interactive way
- 🧠 **Conversational Memory**: Maintains chat history for contextual follow-up questions
- 🤖 **Local LLM Integration**: Powered by Ollama for private, on-device processing
- 📚 **Context-Aware Responses**: Provides answers based on relevant chunks from your PDF
- 💾 **Persistent Vector Database**: Caches embeddings with automatic updates when PDF changes
- ⚡ **Streaming Responses**: Real-time answer generation with streaming support
- 📊 **Performance Metrics**: Real-time tracking of response time, token usage, and compression savings
- 🎛️ **Dynamic Model Selection**: Choose from available Ollama models for different use cases
- ⚙️ **Performance Tuning**: Adjustable CPU threads, temperature, and other parameters
- 📤 **Chat Export**: Download conversation history as text files
- 🖥️ **System Information**: Displays CPU cores and recommended thread settings

## How It Works

1. **PDF Reading**: Extracts text from all pages of a PDF document
2. **Advanced Chunking**: Uses recursive token-based splitting with intelligent separators (paragraphs, sentences) to divide text into chunks of up to 250 tokens with 50 token overlap for better context preservation
3. **Embedding Generation**: Creates vector embeddings for each chunk using Ollama's nomic-embed-text model (L2-normalized)
4. **FAISS Indexing**: Stores embeddings in a FAISS vector index (IndexFlatIP) for fast similarity search
5. **HYDE-Enhanced Search**: Generates a hypothetical answer to augment the search query, then retrieves the top 12 candidate chunks using FAISS similarity search
6. **Intelligent Re-ranking**: FlashRank re-ranks the retrieved chunks using the MS-MARCO-MiniLM model for better relevance
7. **Context Compression**: Selects the most relevant sentences from the top 3 re-ranked chunks based on query keyword matching to reduce token usage while preserving key information
8. **Conversational Context**: Incorporates the last 5 messages from chat history for contextual follow-up questions
9. **Answer Generation**: Uses the selected LLM to generate accurate, cited answers based on the compressed context

## Vector Database Persistence

The application automatically caches the vector database and embedding vectors:
- **Location**: Stored in the `db/` folder
- **Files**: 
  - `index.faiss`: FAISS vector index
  - `metadata.pkl`: Chunk metadata and PDF hash
- **Auto-Update**: When the PDF file changes (detected via SHA-256 hash), the database is automatically regenerated
- **Override Option**: Set `OVERRIDE_DB = True` to force database regeneration

## Requirements

- Python 3.7 or higher
- [Ollama](https://ollama.ai) installed and running locally
- Required Python packages (see installation section):
  - **streamlit**: For the web interface
  - **pdfplumber**: For PDF text extraction
  - **ollama**: For local LLM and embedding model access
  - **numpy**: For vector operations and similarity calculations
  - **faiss-cpu**: For efficient vector similarity search and indexing
  - **flashrank**: For intelligent result re-ranking
  - **psutil**: For system information and thread management
  - **tiktoken**: For token counting and chunking
  - **hashlib**: For PDF hashing (built-in)
  - **pickle**: For metadata serialization (built-in)
  - **logging**: For logging (built-in)
  - **time**: For performance timing (built-in)
  - **subprocess**: For Ollama server management (built-in)
  - **atexit**: For cleanup (built-in)
  - **re**: For regex operations (built-in)

## Installation

### 1. Install Python Dependencies

First, ensure you have Python 3.7+ installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

The following packages will be installed:
- **streamlit**: For building the web interface
- **pdfplumber**: For extracting text from PDF documents
- **ollama**: For accessing local LLM and embedding models running on Ollama
- **numpy**: For efficient vector operations and mathematical calculations
- **faiss-cpu**: For fast vector similarity search and FAISS indexing
- **flashrank**: For intelligent re-ranking of search results using MS-MARCO-MiniLM model
- **psutil**: For retrieving system information like CPU cores
- **tiktoken**: For accurate token counting in text processing

> **Note**: If you have a compatible GPU, use `faiss-gpu` instead of `faiss-cpu` for faster performance

### 2. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai)

### 3. Pull Required Models

Once Ollama is installed, pull the required models:

```bash
# Pull the embedding model (required for semantic search)
ollama pull nomic-embed-text

# Pull the reasoning/generation model (required for answer generation and HyDE)
ollama pull gemma3:1b
```

### 4. Start Ollama Server

Start the Ollama service (runs on `localhost:11434` by default):

```bash
ollama serve
```

Or if Ollama is installed as a service, it will start automatically on system startup.

> **Note**: The application will automatically attempt to start Ollama if it's not running when you launch the app.

## Usage

### Run the Application

To start the Streamlit web application, run:

```bash
streamlit run main.py
```

This will launch the app in your default web browser at `http://localhost:8501`.

### Using the Web Interface

1. **Upload a PDF**: Use the file uploader in the sidebar to select and upload your PDF document
2. **Index the Document**: Click the "🚀 Index & Start" button to process and index the PDF. This may take a few moments for large documents.
3. **Start Chatting**: Once indexed, you can ask questions about the PDF content in the chat input box.
4. **View Responses**: The AI will provide answers based on the document, with sources cited by page numbers and real-time metrics displayed.
5. **Clear Chat History**: Use the "🗑️ Clear Chat History" button in the sidebar to reset the conversation.
6. **Model Selection**: Choose different AI models from the dropdown in the sidebar for varied responses.
7. **Performance Tuning**: Adjust CPU threads and temperature settings for optimal performance.
8. **Export Chat**: Download the conversation history as a text file using the footer button.

### Example Queries

- "What is the main topic of this document?"
- "Explain the key concepts mentioned in this PDF"
- "What does it say about [specific topic]?"
- "Give me examples from the document"
- "Summarize page 3" (for page-specific queries)
- Follow-up: "Can you elaborate on that point?" (uses conversational memory)

## Configuration

You can customize the behavior by modifying these parameters in `main.py`:

### Core Parameters
```python
# Legacy parameters (not used in current token-based chunking)
CHUNK_SIZE = 250           # Not used - kept for compatibility
OVERLAP_SIZE = 50          # Not used - kept for compatibility

# Active parameters
EMBED_MODEL = "nomic-embed-text:latest"  # Embedding model for semantic search
DEFAULT_THINKING_MODEL = "gemma3:1b"      # Default LLM model for generating answers
HYDE_MODEL = "gemma3:1b"                  # Model for generating hypothetical answers
BATCH_SIZE = 32                           # Batch size for embedding generation
TOP_K = 3                                 # Number of top re-ranked chunks to use
```

### Database & Performance Parameters
```python
DB_FOLDER = "db"            # Folder to store vector database and metadata
OVERRIDE_DB = False         # Set to True to force regeneration of the vector database
```

### Streaming & Connection Parameters
```python
STREAM = True               # Enable streaming responses for real-time output
KEEP_ALIVE = '1h'          # How long to keep the Ollama model in memory (reduces reload time)
```

### Chunking Parameters
```python
MAX_SENTENCES = 8           # Maximum sentences to select in context compression
MAX_TOKENS = 250            # Maximum tokens per chunk in recursive splitting
OVERLAP_TOKENS = 50         # Token overlap between chunks
TOKENIZER_CACHE = "tokenizer_cache"  # Cache directory for tiktoken
FAISS_SEARCH_K = 12         # Number of candidates to retrieve before re-ranking
```

### Advanced Tuning
- **Reduce BATCH_SIZE** if you encounter memory issues during embedding generation
- **Increase MAX_TOKENS** for longer context windows but slower processing
- **Adjust TOP_K** to balance between context breadth and answer relevance
- **Modify MAX_SENTENCES** to control the level of context compression (higher = more context, more tokens)
- **Change FAISS_SEARCH_K** to retrieve more candidates for potentially better re-ranking (higher = better quality, slower)
- **Adjust temperature** in the UI for more creative (higher) or focused (lower) responses
- **Increase CPU threads** for faster processing if you have available cores (monitor system responsiveness)

## Troubleshooting

### Ollama Connection Error
- Make sure Ollama service is running: `ollama serve`
- Verify it's accessible at `localhost:11434`
- Check if Ollama is properly installed on your system

### Missing Models
- Ensure you've pulled both required models:
  ```bash
  ollama pull nomic-embed-text:latest
  ollama pull gemma3:1b
  ```
- Check installed models with: `ollama list`

### PDF Not Found
- Check that the PDF file exists in the project directory
- Verify the `PDF_PATH` variable matches your filename exactly
- Use absolute path if PDF is in a different location

### Memory Issues
- **Embedding Generation**: Reduce `BATCH_SIZE` if you run out of memory during processing
- **Chunk Processing**: Reduce `MAX_TOKENS` for faster processing with larger PDFs
- **LLM Inference**: Reduce `KEEP_ALIVE` value to unload models more frequently
- **FAISS Indexing**: Large PDFs may require more RAM for indexing

### Vector Database Issues
- **Force Regeneration**: Set `OVERRIDE_DB = True` to rebuild the vector database
- **Clear Database**: Delete the `db/` folder to start fresh
- **Missing Index**: If you get "index.faiss not found" error, the database will be regenerated automatically
- **Outdated Database**: The application automatically detects PDF changes via SHA-256 hash and re-indexes

### Slow Performance
- **First Run**: Initial embedding generation and FAISS indexing may take time (depending on PDF size)
- **GPU Acceleration**: Use `faiss-gpu` instead of `faiss-cpu` if you have a compatible NVIDIA GPU
- **Model Size**: Consider using smaller models if latency is critical
- **Batch Size**: Increase `BATCH_SIZE` for faster batch processing (if memory allows)

### Re-ranking Results Not Showing
- Ensure FlashRank is properly installed: `pip install flashrank`
- The re-ranking model will be downloaded on first use to `/model_cache` (or custom cache directory)
- Internet connection is required for initial model download

## Project Structure

```
chatPDF/
├── main.py              # Main application script with Streamlit interface
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── .gitignore           # Git ignore file for excluding temporary files
├── db/                  # Vector database storage (auto-created)
│   ├── index.faiss      # FAISS vector index
│   └── metadata.pkl     # Chunk metadata and PDF hash
├── model_cache/         # Model cache for FlashRank and other models
├── tokenizer_cache/     # Cache for tiktoken tokenizer
└── temp/                # Temporary files for uploaded PDFs
```

## System Requirements

- **RAM**: Minimum 4GB (8GB+ recommended)
- **Storage**: Space for Ollama models (~5-10GB)
- **Network**: Internet connection for initial model download
- **Processor**: Multi-core processor recommended

## License

This project is open source and available for personal and educational use.

## Future Enhancements

- ✅ **Advanced Chunking**: Implemented recursive token-based splitting with intelligent separators
- ✅ **HYDE Integration**: Added Hypothetical Document Embeddings for improved search
- ✅ **Context Compression**: Implemented keyword-based context compression to reduce token usage
- ✅ **Dynamic Model Selection**: Added support for choosing from available Ollama models
- ✅ **File Upload Support**: Implemented PDF upload through Streamlit interface
- 🔄 **Semantic Chunking**: Consider implementing semantic chunking for even better context preservation
- 📚 **Multiple PDF Support**: Support for multiple PDF files simultaneously
- 📤 **Enhanced Export**: Export chat history in various formats (JSON, Markdown)
- 🎯 **Custom Model Fine-tuning**: Allow fine-tuning models on specific document types
- 📝 **Document Summarization**: Automatic document summarization features
- 🌍 **Multi-language Support**: Support for non-English documents
- 🔍 **Advanced Search Filters**: Date ranges, author filters, and content type filtering

## Contributing

Feel free to fork this project, make improvements, and submit pull requests.

## Support

For issues or questions, please check the troubleshooting section or create an issue in the repository.

