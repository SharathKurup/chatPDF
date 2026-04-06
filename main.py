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

st.set_page_config(page_title="Chat PDF", layout="wide")
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
    if "user_threads" not in st.session_state:
        st.session_state.user_threads = get_safe_threads()
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.1
    if "thinking_model" not in st.session_state:
        st.session_state.thinking_model = DEFAULT_THINKING_MODEL
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.title("Document Settings")
        uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_file:
            tmp_path = os.path.join(TEMP_PATH, uploaded_file.name)
            if not os.path.exists(TEMP_PATH): os.mkdir(TEMP_PATH)

            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"Uploaded: {uploaded_file.name}")

            if st.button("Index & Start"):
                with st.spinner(text="Analyzing document...", show_time=True):
                    idx, metadata = build_pipeline(tmp_path,st.session_state.thinking_model)
                    st.session_state.index = idx
                    st.session_state.metadata = metadata
                    st.success(f"Ready to Chat!")

            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

            available_models = get_ollama_models()
            st.session_state.thinking_model = st.selectbox(
                "Choose model:",
                available_models,
                index=available_models.index(st.session_state.thinking_model) if st.session_state.thinking_model in available_models else 0
            )

            st.subheader("⚙️ Performance Tuning")
            st.session_state.user_threads = st.slider(
                "CPU Threads", 
                1, 
                psutil.cpu_count(), 
                st.session_state.user_threads
            )
            st.session_state.temperature = st.slider(
                "Creativity (Temp)", 
                0.0, 
                1.0, 
                st.session_state.temperature
            )

    st.title("Chat with your PDF")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something about the PDF..."):
        if "index" not in st.session_state:
            st.error("Please upload and index a PDF first!")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                # better for page specific questions like "what is on page 3".
                # issues starts from historical data, hence removed
                # optimize latter

                # new_page_match = re.search(r"page\s+(\d+)", prompt.lower())
                # if new_page_match:
                #     st.session_state.current_page = int(new_page_match.group(1))
                # current_page = st.session_state.get("current_page")
                # results = search_with_rerank(prompt, st.session_state.index, st.session_state.metadata,forced_page=current_page)

                results = search_with_rerank(prompt, st.session_state.index, st.session_state.metadata)

                start_gen = time.perf_counter()

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

                col1, col2, col3 = st.columns(3)
                col1.metric("Time Taken", f"{gen_time:.2f}s")
                col2.metric("Tokens Used", f"{stats['compressed_tokens']}")
                # col3.metric("Tokens Saved", f"{stats['saved_tokens']}", delta_color="normal")

                response_placeholder.markdown(full_response)
                logger.info(f"Response generated in {gen_time:.2f}s, tokens: {stats['compressed_tokens']}")

                pages = sorted({res["page"] for res in results})
                st.caption(f"Sources: Pages {', '.join(map(str, pages))}")

            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main()

