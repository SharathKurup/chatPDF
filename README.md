# chatPDF

A conversational AI application that allows you to chat with PDF documents using local language models and semantic search.

## Features

- 📄 **PDF Processing**: Extract and process text from PDF documents
- 🔍 **Semantic Search**: Uses embeddings to find relevant content from your PDFs
- 💬 **Conversational Interface**: Ask questions about your PDF in a natural, interactive way
- 🤖 **Local LLM Integration**: Powered by Ollama for private, on-device processing
- 📚 **Context-Aware Responses**: Provides answers based on relevant chunks from your PDF

## How It Works

1. **PDF Reading**: Extracts text from all pages of a PDF document
2. **Chunking**: Divides the text into overlapping chunks for better context preservation
3. **Embedding Generation**: Creates vector embeddings for each chunk using Ollama's embedding model
4. **Semantic Search**: When you ask a question, it finds the most relevant chunks using similarity search
5. **Answer Generation**: Uses an LLM to generate accurate answers based on the retrieved context

## Requirements

- Python 3.7 or higher
- [Ollama](https://ollama.ai) installed and running locally
- Required Python packages (see installation section)

## Installation

### 1. Install Python Dependencies

First, ensure you have Python installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

The following packages will be installed:
- **pdfplumber**: For PDF text extraction
- **ollama**: For local LLM and embedding model access
- **numpy**: For vector operations and similarity calculations

### 2. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai)

### 3. Pull Required Models

Once Ollama is installed, you need to pull the required models:

```bash
# Pull the embedding model
ollama pull nomic-embed-text

# Pull the reasoning/generation model
ollama pull llama3.1
```

### 4. Start Ollama Server

Start the Ollama service (it typically runs on `localhost:11434`):

```bash
ollama serve
```

Or if Ollama is installed as a service, it may start automatically.

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

```python
CHUNK_SIZE = 1024           # Size of text chunks
OVERLAP_SIZE = 200          # Overlap between consecutive chunks
EMBED_MODEL = "nomic-embed-text"  # Embedding model
THINKING_MODEL = "llama3.1:latest"  # LLM model for answers
BATCH_SIZE = 32             # Batch size for embedding generation
TOP_K = 3                   # Number of top chunks to retrieve
```

## Troubleshooting

### Ollama Connection Error
- Make sure Ollama service is running: `ollama serve`
- Verify it's accessible at `localhost:11434`

### Missing Models
- Ensure you've pulled both required models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.1
  ```

### PDF Not Found
- Check that the PDF file exists in the project directory
- Verify the `PDF_PATH` variable matches your filename exactly

### Memory Issues
- Reduce `BATCH_SIZE` if you run out of memory
- Reduce `CHUNK_SIZE` for faster processing with larger PDFs

## Project Structure

```
chatPDF/
├── main.py              # Main application script
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── AI Module.pdf       # Your PDF document
```

## System Requirements

- **RAM**: Minimum 4GB (8GB+ recommended)
- **Storage**: Space for Ollama models (~5-10GB)
- **Network**: Internet connection for initial model download
- **Processor**: Multi-core processor recommended

## License

This project is open source and available for personal and educational use.

## Future Enhancements

- Support for multiple PDF files
- Persistent vector database
- Web interface
- Export chat history
- Custom model selection
- Document summarization

## Contributing

Feel free to fork this project, make improvements, and submit pull requests.

## Support

For issues or questions, please check the troubleshooting section or create an issue in the repository.

