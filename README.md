# chatPDF

A conversational AI application that allows you to chat with PDF documents using local language models, semantic search, and intelligent result re-ranking.

## Features

- 📄 **PDF Processing**: Extract and process text from PDF documents
- 🔍 **Semantic Search with FAISS**: Fast vector similarity search using FAISS for efficient retrieval
- 🎯 **Intelligent Re-ranking**: Uses FlashRank to re-rank search results for better relevance
- 💬 **Conversational Interface**: Ask questions about your PDF in a natural, interactive way
- 🤖 **Local LLM Integration**: Powered by Ollama for private, on-device processing
- 📚 **Context-Aware Responses**: Provides answers based on relevant chunks from your PDF
- 💾 **Persistent Vector Database**: Caches embeddings with automatic updates when PDF changes
- ⚡ **Streaming Responses**: Real-time answer generation with streaming support

## How It Works

1. **PDF Reading**: Extracts text from all pages of a PDF document
2. **Chunking**: Divides the text into overlapping chunks for better context preservation
3. **Embedding Generation**: Creates vector embeddings for each chunk using Ollama's embedding model (L2-normalized)
4. **FAISS Indexing**: Stores embeddings in a FAISS vector index (IndexFlatIP) for fast similarity search
5. **Semantic Search**: When you ask a question, it retrieves the top 10 most relevant chunks using FAISS similarity search
6. **Re-ranking**: FlashRank re-ranks the retrieved chunks using the MS-MARCO-MiniLM model for improved relevance
7. **Answer Generation**: Uses an LLM to generate accurate answers based on the top re-ranked chunks

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
  - **pdfplumber**: For PDF text extraction
  - **ollama**: For local LLM and embedding model access
  - **numpy**: For vector operations and similarity calculations
  - **faiss-cpu**: For efficient vector similarity search and indexing
  - **flashrank**: For intelligent result re-ranking

## Installation

### 1. Install Python Dependencies

First, ensure you have Python 3.7+ installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

The following packages will be installed:
- **pdfplumber**: For extracting text from PDF documents
- **ollama**: For accessing local LLM and embedding models running on Ollama
- **numpy**: For efficient vector operations and mathematical calculations
- **faiss-cpu**: For fast vector similarity search and FAISS indexing
- **flashrank**: For intelligent re-ranking of search results using MS-MARCO-MiniLM model

> **Note**: If you have a compatible GPU, use `faiss-gpu` instead of `faiss-cpu` for faster performance

### 2. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai)

### 3. Pull Required Models

Once Ollama is installed, pull the required models:

```bash
# Pull the embedding model (required for semantic search)
ollama pull nomic-embed-text

# Pull the reasoning/generation model (required for answer generation)
ollama pull llama3.1
```

### 4. Start Ollama Server

Start the Ollama service (runs on `localhost:11434` by default):

```bash
ollama serve
```

Or if Ollama is installed as a service, it will start automatically on system startup.

## Usage

### Prepare Your PDF

Place your PDF file in the project directory and update the `PDF_PATH` variable in `main.py`:

```python
PDF_PATH = "your_pdf_file.pdf"
```

### Run the Application

```bash
python main.py
```

### Interactive Chat

Once the application starts:

1. The PDF will be automatically processed and indexed
2. You'll see the prompt: `You - `
3. Type your questions about the PDF content
4. The AI will search for relevant sections and provide answers
5. Type `exit` to quit the application

### Example Queries

- "What is the main topic of this document?"
- "Explain the key concepts mentioned in this PDF"
- "What does it say about [specific topic]?"
- "Give me examples from the document"

## Configuration

You can customize the behavior by modifying these parameters in `main.py`:

### Core Parameters
```python
CHUNK_SIZE = 1024           # Size of text chunks in characters
OVERLAP_SIZE = 200          # Overlap between consecutive chunks for context preservation
EMBED_MODEL = "nomic-embed-text:latest"  # Embedding model for semantic search
THINKING_MODEL = "llama3.1:latest"  # LLM model for generating answers
BATCH_SIZE = 32             # Batch size for embedding generation (reduce if out of memory)
TOP_K = 3                   # Number of top re-ranked chunks to use for answer generation
```

### Database & Performance Parameters
```python
DB_FOLDER = "db"            # Folder to store vector database and metadata
OVERRIDE_DB = False         # Set to True to force regeneration of the vector database
PDF_PATH = "AI Module.pdf"  # Path to your PDF file
```

### Streaming & Connection Parameters
```python
STREAM = True               # Enable streaming responses for real-time output
KEEP_ALIVE = '1h'          # How long to keep the Ollama model in memory (reduces reload time)
```

### Advanced Tuning
- **Reduce BATCH_SIZE** if you encounter memory issues during embedding generation
- **Increase CHUNK_SIZE** for longer context windows but slower processing
- **Adjust TOP_K** to balance between context breadth and answer relevance
- **Modify OVERLAP_SIZE** to control context continuity between chunks

## Troubleshooting

### Ollama Connection Error
- Make sure Ollama service is running: `ollama serve`
- Verify it's accessible at `localhost:11434`
- Check if Ollama is properly installed on your system

### Missing Models
- Ensure you've pulled both required models:
  ```bash
  ollama pull nomic-embed-text:latest
  ollama pull llama3.1:latest
  ```
- Check installed models with: `ollama list`

### PDF Not Found
- Check that the PDF file exists in the project directory
- Verify the `PDF_PATH` variable matches your filename exactly
- Use absolute path if PDF is in a different location

### Memory Issues
- **Embedding Generation**: Reduce `BATCH_SIZE` if you run out of memory during processing
- **Chunk Processing**: Reduce `CHUNK_SIZE` for faster processing with larger PDFs
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
- The re-ranking model will be downloaded on first use to `/tmp` (or custom cache directory)
- Internet connection is required for initial model download

## Project Structure

```
chatPDF/
├── main.py              # Main application script
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── AI Module.pdf       # Your PDF document
├── db/                 # Vector database storage (auto-created)
│   ├── index.faiss     # FAISS vector index
│   └── metadata.pkl    # Chunk metadata and PDF hash
├── tmp/                # Temporary files and model cache
└── __pycache__/        # Python bytecode cache
```

## System Requirements

- **RAM**: Minimum 4GB (8GB+ recommended)
- **Storage**: Space for Ollama models (~5-10GB)
- **Network**: Internet connection for initial model download
- **Processor**: Multi-core processor recommended

## License

This project is open source and available for personal and educational use.

## Future Enhancements

- Web interface
- Support for multiple PDF files
- Persistent vector database
- Export chat history
- Custom model selection
- Document summarization

## Contributing

Feel free to fork this project, make improvements, and submit pull requests.

## Support

For issues or questions, please check the troubleshooting section or create an issue in the repository.

