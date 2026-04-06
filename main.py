import os
import sys
import numpy as np
import ollama
import pdfplumber
import faiss
import time
import pickle
import hashlib
from flashrank import Ranker, RerankRequest
import logging
import streamlit as st
import re
import psutil
import tiktoken
import subprocess
import atexit

ollama_env = os.environ.copy()
ollama_env["OLLAMA_FLASH_ATTENTION"] = "1"
ollama_env["OLLAMA_KV_CACHE_TYPE"] = "q8_0" # Another 2026 speed optimization

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

PDF_PATH = os.environ.get("PDF_PATH", "AI Module.pdf")
CHUNK_SIZE = 250
OVERLAP_SIZE = 50
EMBED_MODEL = "nomic-embed-text:latest"
DEFAULT_THINKING_MODEL = "gemma3:1b"
HYDE_MODEL = "gemma3:1b"
BATCH_SIZE = 32
TOP_K = 3
STREAM = True
KEEP_ALIVE = '1h'
DB_FOLDER = "db"
OVERRIDE_DB = False
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2" # https://huggingface.co/prithivida/flashrank/tree/main
CACHE_DIR = "./model_cache"
TEMP_PATH = "./temp"
MAX_SENTENCES = 8
MAX_TOKENS = 250
OVERLAP_TOKENS = 50
TOKENIZER_CACHE = "tokenizer_cache"
FAISS_SEARCH_K = 12

_ollama_process = None

def is_ollama_running():
    try:
        ollama.list()
        return True
    except Exception:
        return False

def start_ollama_server():
    global _ollama_process
    if is_ollama_running(): return True
    try:
        env = os.environ.copy()
        env["OLLAMA_FLASH_ATTENTION"] = "1"
        _ollama_process = subprocess.Popen(
            ['ollama', 'serve'], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        for i in range(20):
            if is_ollama_running():
                logger.info("Ollama started successfully.")
                return True
            time.sleep(1)
        return False
    except Exception as e:
        logger.error(f"Ollama startup error: {e}")
        return False

def cleanup_ollama():
    if _ollama_process:
        _ollama_process.terminate()
        logger.info("Ollama process terminated.")

TIKTOKEN_CACHE_DIR = os.path.join(os.getcwd(), TOKENIZER_CACHE)
if not os.path.exists(TIKTOKEN_CACHE_DIR):
    os.makedirs(TIKTOKEN_CACHE_DIR)
os.environ["TIKTOKEN_CACHE_DIR"] = TIKTOKEN_CACHE_DIR

# https://github.com/PrithivirajDamodaran/FlashRank

def readpdf(pdf_file):
    logger.info(f"Reading PDF: {pdf_file}")
    all_texts = []

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                all_texts.append((i + 1, text))
    except FileNotFoundError:
        logger.error(f"PDF not found: {pdf_file}")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Failed to read PDF: {exc}")
        sys.exit(1)

    return all_texts

def get_safe_threads():
    # Returns logical cores. For a 16-core CPU, this returns 16.
    # We subtract 1 or 2 to keep the OS and Streamlit responsive.
    cores = psutil.cpu_count(logical=True)
    return max(1, cores - 2)

st.set_page_config(
    page_title="Chat PDF - AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)
if not is_ollama_running():
    with st.spinner("Initializing Ollama..."):
        if not start_ollama_server():
            st.error("Failed to start Ollama."); st.stop()
        atexit.register(cleanup_ollama)
tokenizer = tiktoken.get_encoding("cl100k_base")

def get_token_length(text):
    return len(tokenizer.encode(text))

def generate_advanced_chunks(page_content,page_num):
    search_chunks = generate_chunks_recursive_tokens(page_content,page_num)
    # search_chunks = generate_chunks_recursive(page_content, page_num, 1024,200)
    for chunk in search_chunks:
        chunk["text"] = f"[Page {page_num}] {chunk['text']}"
        chunk["full_context"] = page_content

    return search_chunks

def generate_chunks_recursive_tokens(text,page_num):
    if not text: 
        logger.debug(f"Empty text provided for page {page_num}")
        return []
    chunks = []
    paragraphs = text.split("\n\n")
    logger.debug(f"Page {page_num}: Split into {len(paragraphs)} paragraphs")
    current_chunk = []
    current_tokens=0

    for paragraph in paragraphs:
        paragraph_tokens = get_token_length(paragraph)
        if paragraph_tokens > MAX_TOKENS:
            sentences=re.split(r'(?<=[.!?]) +', paragraph)
            for sentence in sentences:
                sentence_token = get_token_length(sentence)
                if current_tokens + sentence_token > MAX_TOKENS:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({"text": chunk_text, "page": page_num})
                    current_chunk = current_chunk[-2:] if len(current_chunk) > 2 else []
                    current_tokens=sum(get_token_length(cur_chunk) for cur_chunk in current_chunk)
                current_chunk.append(sentence)
                current_tokens+=sentence_token
        else:
            if current_tokens + paragraph_tokens > MAX_TOKENS:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({"text": chunk_text, "page": page_num})
                current_chunk=[]
                current_tokens=0
            current_chunk.append(paragraph)
            current_tokens+=paragraph_tokens

    if current_chunk:
        chunks.append({"text": "\n\n".join(current_chunk), "page": page_num})

    logger.debug(f"Page {page_num}: Created {len(chunks)} chunks from {len(paragraphs)} paragraphs")
    return chunks

def generate_embeddings_batch(texts):
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Starting embeddings batch processing for {len(texts)} texts in {total_batches} batches (batch size: {BATCH_SIZE})")
    
    for batch_idx, i in enumerate(range(0, len(texts), BATCH_SIZE), 1):
        batch_texts = texts[i:i+BATCH_SIZE]
        try:
            logger.debug(f"Processing batch {batch_idx}/{total_batches} with {len(batch_texts)} texts")
            response = ollama.embed(model=EMBED_MODEL, input=batch_texts)
            all_embeddings.extend(response["embeddings"])
        except Exception as exc:
            logger.error(f"Embedding request failed for batch {batch_idx}: {exc}")
            raise RuntimeError(f"Embedding request failed: {exc}") from exc
    
    logger.info(f"Completed embedding generation: {len(all_embeddings)} embeddings created")
    return all_embeddings

def generate_answer(query, results, chat_history,threads,temp, thinking_model):
    options = {
        "num_thread": threads,
        "temperature": temp
        # "num_ctx": 8192  # Limits context window to keep it fast
    }

    raw_context_tokens = sum(get_token_length(res['full_context']) for res in results)
    logger.debug(f"Query: {query[:100]}..." if len(query) > 100 else f"Query: {query}")
    page_match = re.search(r"page\s+(\d+)", query.lower())

    context_parts = []
    for res in results:
        # If user asked for a specific page, send the WHOLE page content
        # Otherwise, use the keyword compression for general questions
        if page_match:
            compressed_text = res['full_context']
        else:
            compressed_text = compress_context(query, res['full_context'])

        context_parts.append(
            f"--- START OF PAGE {res['page']} ---\n{compressed_text}\n--- END OF PAGE {res['page']} ---")
    for res in results:
        compressed_text = compress_context(query, res['full_context'])
        context_parts.append(
            f"--- START OF PAGE {res['page']} ---\n{compressed_text}\n--- END OF PAGE {res['page']} ---")

    compressed_context_text = "\n\n".join(context_parts).strip()
    compressed_tokens = get_token_length(compressed_context_text)

    stats = {
        "raw_tokens": raw_context_tokens,
        "compressed_tokens": compressed_tokens,
        "saved_tokens": raw_context_tokens - compressed_tokens
    }
    logger.debug(f"Token compression stats - Raw: {raw_context_tokens}, Compressed: {compressed_tokens}, Saved: {stats['saved_tokens']} ({100*stats['saved_tokens']/raw_context_tokens if raw_context_tokens > 0 else 0:.1f}%)")

    history_text = "\n".join([f"{history['role']}: {history['content']}" for history in chat_history[-5:]])

    context_text = "\n\n".join(context_parts).strip()

    prompt = f"""
You are a professional research assistant. 
Review the ENTIRE context provided below and provide a comprehensive, 
detailed response covering all sections mentioned.

STRICT RULE: Your answer MUST be based ONLY on the provided Context. 
Do NOT use outside knowledge. 
If the information is not in the Context, say "I don't know based on this page."

Instructions:
1. Every time you state a fact from the context, cite the page number immediately after the sentence in brackets, like this: [Page X].
2. If the answer isn't in the context, clearly state that you don't know.
3. Keep your response structured and easy to read.

IMPORTANT:
If the question asks about a specific page, ONLY use content from that page, previous page and next page.
Ignore all other pages.

Context:
{context_text}

Conversation History:
{history_text}

Question:
{query}

Answer:
"""
    # if STREAM:
    return ollama.generate(
        model=thinking_model,
        prompt=prompt,
        stream=STREAM,
        keep_alive=KEEP_ALIVE,
        options=options,
    ),stats
    # response = ollama.generate(model=THINKING_MODEL, prompt=prompt, keep_alive=KEEP_ALIVE)
    # return response["response"],stats

def generate_hypothetical_answer(query):
    """HyDE: Generates a brief fake answer to improve vector search."""
    prompt = f"Write a 2-sentence technical summary answering: {query}"
    # Use a fast call to Ollama (non-streaming for speed)
    response = ollama.generate(model=HYDE_MODEL, prompt=prompt, stream=False)
    return response['response']

def verify_normalized_embedding(embeddings):
    """
        How to Check if the Output is Normalized
        In the context of embeddings, "normalization" almost always refers to Vector Normalization (L2),
        which ensures the embedding has a magnitude (length) of exactly 1.
        You can check this mathematically using Python.
        If the sum of the squares of all numbers in the embedding vector equals approximately 1.0, it is normalized.
    """
    embeddings_array = np.array(embeddings)
    norms = np.linalg.norm(embeddings_array, axis=1)

    all_normalized = np.all(np.isclose(norms, 1.0))

    logger.debug(f"Vector Magnitude: {norms}")

    if all_normalized:
        logger.info("The embedding is L2 normalized.")
    else:
        logger.warning("The embedding is NOT normalized.")

def check_ollama_status(thinking_model):
    try:
        response = ollama.list()
        # downloaded_models=[model.get("model") for model in response.get("models",[])]
        downloaded_models = [m.model for m in response.models]
        logger.debug(f"Available Ollama models: {downloaded_models}")
        is_model_missing = [model for model in [EMBED_MODEL,DEFAULT_THINKING_MODEL,thinking_model] if model not in downloaded_models]

        if not is_model_missing:
            logger.info(f"All required models available: {[EMBED_MODEL, DEFAULT_THINKING_MODEL, thinking_model]}")
            return True
        else:
            logger.warning(f"Missing models: {is_model_missing}")
            return False
    except Exception as exc:
        logger.error(f"Failed to check Ollama status: {exc}")
        return False

def save_vector_db(index,metadata,current_hash):
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
    faiss.write_index(index, os.path.join(DB_FOLDER, "index.faiss"))
    data = {"metadata": metadata,"pdf_hash": current_hash}
    with open(os.path.join(DB_FOLDER, "metadata.pkl"), "wb") as f:
        pickle.dump(data, f)
    logger.info("Saved vector database and PDF Hash value.")

def load_vector_db():
    if os.path.exists(DB_FOLDER):
        index = faiss.read_index(os.path.join(DB_FOLDER, "index.faiss"))
        with open(f"{DB_FOLDER}/metadata.pkl", "rb") as f:
            data = pickle.load(f)
        logger.info("Database loaded successfully.")
        return index, data.get("metadata"), data.get("pdf_hash")
    return None, None, None

def calculate_pdf_hash(pdf_file):
    logger.debug(f"Calculating hash for PDF: {pdf_file}")
    sha256_hash = hashlib.sha256()
    try:
        with open (pdf_file, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        hash_value = sha256_hash.hexdigest()
        logger.debug(f"PDF hash calculated: {hash_value}")
        return hash_value
    except Exception as exc:
        logger.error(f"Failed to calculate PDF hash: {exc}")
        raise

@st.cache_resource
def get_ranker():
    return Ranker(model_name=RERANK_MODEL,cache_dir=CACHE_DIR)


def search_with_rerank(query, index, text_metadata):
    ranker = get_ranker()
    query = query.strip()
    if not query:
        logger.warning("Empty query provided to search_with_rerank")
        return []

    page_match = re.search(r"page\s+(\d+)", query.lower())
    target_page = int(page_match.group(1)) if page_match else None

    if target_page:
        start_time = time.perf_counter()

        # Strict filtering (no semantic noise)
        candidates = [
            item for item in text_metadata
            if item["page"] == target_page
        ]

        if not candidates:
            logger.warning(f"No candidates found for page {target_page}")
            return []

        # Prepare reranker input
        rerank_items = [
            {
                "id": i,
                "text": c["text"],
                "page": c["page"],
                "full_context": c.get("full_context", "")
            }
            for i, c in enumerate(candidates)
        ]

        rerank_request = RerankRequest(query=query, passages=rerank_items)
        results = ranker.rerank(rerank_request)

        end_time = time.perf_counter()
        logger.info(f"[PAGE MODE] Page {target_page} | Time: {end_time - start_time:.2f}s")

        return results  # return full page context

    start_time = time.perf_counter()

    # HyDE (only for semantic queries)
    hypothetical_answer = generate_hypothetical_answer(query)
    search_query = f"{query} {hypothetical_answer}"

    # Embedding
    response = ollama.embed(model=EMBED_MODEL, input=search_query)
    query_vector = np.array(response["embeddings"][0], dtype=np.float32)

    faiss_start = time.perf_counter()
    distances, indices = index.search(query_vector.reshape(1, -1), k=FAISS_SEARCH_K)
    faiss_end = time.perf_counter()

    candidates = [text_metadata[i] for i in indices[0]]

    rerank_items = [
        {
            "id": i,
            "text": c["text"],
            "page": c["page"],
            "full_context": c.get("full_context", "")
        }
        for i, c in enumerate(candidates)
    ]

    rerank_request = RerankRequest(query=query, passages=rerank_items)
    results = ranker.rerank(rerank_request)

    end_time = time.perf_counter()

    logger.info(f"[SEMANTIC MODE] Total Time: {end_time - start_time:.2f}s")
    logger.debug(f"  → FAISS Time: {faiss_end - faiss_start:.2f}s")
    logger.debug(f"  → Candidates: {len(candidates)}")

    logger.debug("\n--- Re-ranker Scores ---")
    for i, res in enumerate(results[:TOP_K]):
        logger.debug(f"Rank {i+1}: Score {res['score']:.4f} (Page {res['page']})")
    logger.debug("------------------------\n")

    return results[:TOP_K]

def compress_context(query, full_text):
    # Split into sentences
    sentences = re.split(r'(?<=[.!?]) +', full_text)
    query_words = set(query.lower().split())

    scored_sentences = []
    for i,s in enumerate(sentences):
        score = sum(1 for word in s.lower().split() if word in query_words)
        scored_sentences.append((score,i , s))

    top_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:MAX_SENTENCES]
    top_sentences = sorted(top_sentences, key=lambda x:x[1])
    compressed = " ".join([s for _,_, s in top_sentences])

    return compressed if compressed else full_text[:1000]  # Fallback to snippet

def build_pipeline(pdf_file,thinking_model):
    logger.info(f"=" * 60)
    logger.info(f"Starting pipeline for PDF: {pdf_file}")
    logger.info(f"=" * 60)
    
    if not check_ollama_status(thinking_model):
        error_msg = f"Required models not available. Please run: ollama pull {EMBED_MODEL} && ollama pull {thinking_model}"
        logger.error(error_msg)
        st.error(error_msg)
        st.stop()

    logger.info("Step 1: Calculating PDF hash...")
    current_hash = calculate_pdf_hash(pdf_file)
    
    logger.info("Step 2: Attempting to load existing vector database...")
    index, metadata,stored_hash = load_vector_db()

    if index and metadata and current_hash == stored_hash and not OVERRIDE_DB:
        logger.info(f"Hash match found (stored: {stored_hash[:16]}...). Using existing database.")
        st.info("Existing database found for this file. Loading...")
        return index, metadata

    logger.info("Hash mismatch or no existing database. Processing new PDF...")
    st.info("New PDF detected. Processing...")
    
    logger.info("Step 3: Reading PDF...")
    texts = readpdf(pdf_file)
    if not texts:
        logger.error("No text found in PDF.")
        st.warning("No text found in PDF.")
        sys.exit(1)

    logger.info(f"Step 4: Creating chunks from {len(texts)} pages...")
    metadata = []
    for page_num, page_content in texts:
        metadata.extend(generate_advanced_chunks(page_content, page_num))

    if not metadata:
        logger.error("No content chunks were created from the PDF.")
        st.warning("No content chunks were created from the PDF.")
        sys.exit(1)

    logger.info(f"Step 5: Generating embeddings for {len(metadata)} chunks...")
    text_data = [item["text"] for item in metadata]
    vectors = generate_embeddings_batch(text_data)
    verify_normalized_embedding(vectors)

    vector_np = np.array(vectors).astype('float32')

    logger.info(f"Step 6: Building FAISS index with dimension {vector_np.shape[1]}...")
    dimension = vector_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vector_np)
    logger.info(f"FAISS index built with {index.ntotal} vectors")

    logger.info("Step 7: Saving vector database...")
    save_vector_db(index, metadata, current_hash)
    
    logger.info(f"=" * 60)
    logger.info(f"Pipeline completed successfully!")
    logger.info(f"Total chunks: {len(metadata)}, Total vectors: {index.ntotal}")
    logger.info(f"=" * 60)

    return index, metadata

def get_ollama_models():
    try:
        response = ollama.list()
        models_list=[]
        for model in response.models:
            models_list.append(model.model)
        logger.debug(f"Retrieved {len(models_list)} available Ollama models")
        return models_list
    except Exception as exc:
        logger.error(f"Failed to retrieve Ollama models: {exc}")
        return [DEFAULT_THINKING_MODEL]

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    .main .block-container {
        background: rgba(30, 30, 30, 0.95);
        color: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem auto;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stSidebar {
        background: rgba(30, 30, 30, 0.95);
        color: white;
        border-radius: 10px;
        margin: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stChatMessage {
        border-radius: 10px;
        margin: 0.5rem 0;
        padding: 1rem;
        background: rgba(45, 45, 45, 0.8) !important;
    }
    .stMetric {
        background: rgba(45, 45, 45, 0.8);
        border-radius: 8px;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stButton>button {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    if "user_threads" not in st.session_state:
        st.session_state.user_threads = get_safe_threads()
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.1
    if "thinking_model" not in st.session_state:
        st.session_state.thinking_model = DEFAULT_THINKING_MODEL
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "index" not in st.session_state:
        st.session_state.index = None
    if "metadata" not in st.session_state:
        st.session_state.metadata = None

    with st.sidebar:
        st.title("Document Assistant")

        st.markdown("---")

        # File Upload Section
        st.subheader("Document Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to start chatting with it"
        )

        if uploaded_file:
            # Reset chat when a new file is selected
            if st.session_state.uploaded_file_name != uploaded_file.name:
                st.session_state.messages = []
                st.session_state.index = None
                st.session_state.metadata = None
                st.session_state.uploaded_file_name = uploaded_file.name

            # File info display
            file_size = len(uploaded_file.getbuffer()) / 1024 / 1024  # MB
            st.success(f"📎 **{uploaded_file.name}** ({file_size:.1f} MB)")

            tmp_path = os.path.join(TEMP_PATH, uploaded_file.name)
            if not os.path.exists(TEMP_PATH): os.mkdir(TEMP_PATH)

            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Action buttons in columns
            index_requested = False
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Index & Start", use_container_width=True):
                    index_requested = True

            with col2:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()

            if index_requested:
                with st.spinner("🔄 Analyzing document..."):
                    progress_bar = st.progress(0)
                    st.text("Step 1/7: Checking models...")
                    progress_bar.progress(14)

                    st.text("Step 2/7: Calculating hash...")
                    progress_bar.progress(28)

                    st.text("Step 3/7: Reading PDF...")
                    progress_bar.progress(42)

                    st.text("Step 4/7: Creating chunks...")
                    progress_bar.progress(56)

                    st.text("Step 5/7: Generating embeddings...")
                    progress_bar.progress(70)

                    st.text("Step 6/7: Building index...")
                    progress_bar.progress(84)

                    st.text("Step 7/7: Saving database...")
                    progress_bar.progress(100)

                    # Reset conversation when indexing a document
                    st.session_state.messages = []
                    idx, metadata = build_pipeline(tmp_path, st.session_state.thinking_model)
                    st.session_state.index = idx
                    st.session_state.metadata = metadata
                    progress_bar.empty()

                st.success("✅ Ready to Chat!")

            st.markdown("---")

            # Model Selection
            st.subheader("AI Model")
            available_models = get_ollama_models()
            current_model = st.session_state.get('thinking_model', DEFAULT_THINKING_MODEL)

            # Find current model index, default to first if not found
            try:
                current_idx = available_models.index(current_model)
            except ValueError:
                current_idx = 0
                if available_models:
                    st.session_state.thinking_model = available_models[0]

            selected_model = st.selectbox(
                "Choose model:",
                available_models if available_models else [DEFAULT_THINKING_MODEL],
                index=current_idx,
                help="Select the AI model for generating responses"
            )
            st.session_state.thinking_model = selected_model

            # Performance Settings
            st.markdown("---")
            st.subheader("Performance Tuning")

            # CPU Threads
            max_threads = psutil.cpu_count()
            safe_threads = get_safe_threads()
            st.session_state.user_threads = st.slider(
                "CPU Threads",
                min_value=1,
                max_value=max_threads,
                value=st.session_state.user_threads,
                help=f"Recommended: {safe_threads} (keeps system responsive)"
            )

            # Temperature
            st.session_state.temperature = st.slider(
                "Creativity (Temperature)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Lower = more focused, Higher = more creative"
            )

            # System Info
            st.markdown("---")
            st.subheader("System Info")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("CPU Cores", f"{max_threads}")
            with col_b:
                st.metric("Safe Threads", f"{safe_threads}")

            # Chat Statistics
            if st.session_state.messages:
                st.markdown("---")
                st.subheader("💬 Chat Stats")
                total_messages = len(st.session_state.messages)
                user_messages = len([m for m in st.session_state.messages if m['role'] == 'user'])
                assistant_messages = len([m for m in st.session_state.messages if m['role'] == 'assistant'])

                col_stats1, col_stats2 = st.columns(2)
                with col_stats1:
                    st.metric("Total Messages", total_messages)
                with col_stats2:
                    st.metric("Q&A Pairs", min(user_messages, assistant_messages))

        # Help section
        with st.expander("❓ Help & Tips", expanded=False):
            st.markdown("""
            ### Getting Started
            1. **Upload PDF**: Use the file uploader above to select your document
            2. **Index Document**: Click "🚀 Index & Start" to process the PDF
            3. **Ask Questions**: Type your questions in the chat input

            ### Tips for Better Results
            - **Be specific**: "What are the main benefits on page 3?" works better than "Tell me about benefits"
            - **Use page numbers**: Mention specific pages when you know them
            - **Ask for summaries**: "Summarize chapter 2" or "Give me an overview"
            - **Context matters**: Reference previous questions for follow-ups

            ### Performance Tuning
            - **CPU Threads**: More threads = faster processing, but may slow your system
            - **Temperature**: Lower = more focused answers, Higher = more creative
            - **Models**: Different models have different strengths

            ### Troubleshooting
            - **No response**: Check if Ollama is running and models are downloaded
            - **Slow performance**: Reduce CPU threads or try a smaller model
            - **Poor answers**: Try rephrasing your question or adjusting temperature
            """)

    # Main Chat Area
    st.title("Chat with your PDF")
    st.markdown("*Ask questions about your uploaded document*")

    # Welcome message when no document is loaded
    if "index" not in st.session_state:
        st.info("👋 **Welcome to Chat PDF!**")
        st.markdown("""
        ### 🚀 Quick Start Guide:
        1. 📄 **Upload** your PDF document using the sidebar
        2. 🚀 **Index** the document by clicking "Index & Start"
        3. 💬 **Chat** with your document using natural language

        ### 💡 Pro Tips:
        - Ask specific questions like *"What are the key findings on page 5?"*
        - Request summaries: *"Summarize the methodology section"*
        - Get detailed explanations: *"Explain the algorithm in detail"*

        ### 🎯 Best Practices:
        - **Page-specific queries**: Mention page numbers when possible
        - **Context-aware**: Reference previous conversations
        - **Iterative refinement**: Follow up with clarifying questions
        """)

        # Feature highlights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🔍 Smart Search\nAI-powered document search with relevance ranking")
        with col2:
            st.markdown("### 📊 Performance Metrics\nReal-time token usage and response time tracking")
        with col3:
            st.markdown("### Modern UI\nClean, responsive interface with a professional look")

        return

    # Chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the PDF...", key="chat_input"):
        if not prompt.strip():
            st.warning("Please enter a question.")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(f"**You:** {prompt}")

        # Generate response
        with chat_container:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                # Search and generate
                with st.spinner("Searching document..."):
                    results = search_with_rerank(prompt, st.session_state.index, st.session_state.metadata)

                if not results:
                    st.error("No relevant information found in the document.")
                    return

                start_gen = time.perf_counter()

                with st.spinner("Generating response..."):
                    stream_response, stats = generate_answer(
                        prompt,
                        results,
                        st.session_state.messages,
                        st.session_state.user_threads,
                        st.session_state.temperature,
                        st.session_state.thinking_model
                    )

                    for chunk in stream_response:
                        full_response += chunk['response']
                        response_placeholder.markdown(full_response + "▌")

                end_gen = time.perf_counter()
                gen_time = end_gen - start_gen

                # Final response
                response_placeholder.markdown(full_response)
                logger.info(f"Response generated in {gen_time:.2f}s, tokens: {stats['compressed_tokens']}")

                # Enhanced metrics display
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Response Time", f"{gen_time:.2f}s")
                with col2:
                    st.metric("Tokens Used", f"{stats['compressed_tokens']:,}")
                with col3:
                    if 'saved_tokens' in stats and stats['raw_tokens'] > 0:
                        savings_pct = (stats['saved_tokens'] / stats['raw_tokens']) * 100
                        st.metric("Token Savings", f"{savings_pct:.1f}%")
                    else:
                        st.metric("Context Quality", "Optimized")
                with col4:
                    st.metric("Sources", f"{len(set(res['page'] for res in results))}")

                # Source pages with better formatting
                pages = sorted(set(res["page"] for res in results))
                if pages:
                    st.caption(f"**Source Pages:** {', '.join(map(str, pages))}")

                    # Add relevance scores for top results
                    with st.expander("🔍 View Relevance Scores", expanded=False):
                        for i, res in enumerate(results[:3], 1):
                            st.write(f"**Rank {i}:** Page {res['page']} (Score: {res.get('score', 'N/A'):.4f})")

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Auto-scroll to bottom (using empty element as anchor)
        st.empty()

    # Footer with additional features
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

    with footer_col1:
        if st.session_state.messages:
            if st.button("Export Chat", help="Download conversation as text file"):
                chat_text = "# Chat PDF Conversation Export\n\n"
                for i, msg in enumerate(st.session_state.messages, 1):
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    chat_text += f"## Message {i} - {role}\n{msg['content']}\n\n"

                st.download_button(
                    label="Download",
                    data=chat_text,
                    file_name="chat_export.txt",
                    mime="text/plain"
                )

    with footer_col2:
        st.caption("**Tips:** Ask specific questions, mention page numbers, or request summaries for better results.")

    with footer_col3:
        st.caption(f"**Model:** {st.session_state.get('thinking_model', 'None loaded')}")

if __name__ == '__main__':
    main()

