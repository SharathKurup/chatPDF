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

# Set this at the very top of your script, before any ollama calls
os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
os.environ["OLLAMA_KV_CACHE_TYPE"] = "q8_0" # Another 2026 speed optimization

logging.getLogger("httpx").setLevel(logging.WARNING)

PDF_PATH = os.environ.get("PDF_PATH", "AI Module.pdf")
CHUNK_SIZE = 1024
OVERLAP_SIZE = 200
EMBED_MODEL = "nomic-embed-text:latest"
THINKING_MODEL = "llama3.1:latest"
BATCH_SIZE = 32
TOP_K = 2
STREAM = True
KEEP_ALIVE = '1h'
DB_FOLDER = "db"
OVERRIDE_DB = False
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2" # https://huggingface.co/prithivida/flashrank/tree/main
CACHE_DIR = "./model_cache"
TEMP_PATH = "./temp"
MAX_SENTENCES = 8

# https://github.com/PrithivirajDamodaran/FlashRank

def readpdf(pdf_file):
    print(f"Reading PDF: {pdf_file}")
    all_texts = []

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                all_texts.append((i + 1, text))
    except FileNotFoundError:
        print(f"PDF not found: {pdf_file}")
        sys.exit(1)
    except Exception as exc:
        print(f"Failed to read PDF: {exc}")
        sys.exit(1)

    return all_texts

def get_safe_threads():
    # Returns logical cores. For a 16-core CPU, this returns 16.
    # We subtract 1 or 2 to keep the OS and Streamlit responsive.
    cores = psutil.cpu_count(logical=True)
    return max(1, cores - 2)

st.set_page_config(page_title="Chat PDF", layout="wide")

def generate_chunks(text, page_num):
    if not text:
        return []

    chunks = []
    i = 0
    while i < len(text):
        end = min(i + CHUNK_SIZE, len(text))
        chunk = text[i:end]

        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space != -1:
                end = i + last_space
                chunk = text[i:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append({"text": chunk, "page": page_num})

        i = end - OVERLAP_SIZE
        if i < 0:
            i = 0
        if i >= len(text) or end == len(text):
            break

    return chunks

def generate_chunks_recursive(text, page_num, chunk_size, overlap_size):
    if not text: return []
    chunks = []
    start = 0
    min_chunk_size = 100  # Avoid tiny chunks at the end of pages

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk_text = text[start:].strip()
            if chunk_text: chunks.append({"text": chunk_text, "page": page_num})
            break

        chunk_slice = text[start:end]

        for separator in ["\n\n", "\n", ". ", " "]:
            last_break = chunk_slice.rfind(separator)
            if last_break != -1:
                if separator == ". ": last_break += 1
                break
        else:
            last_break = chunk_size

        actual_end = start + last_break
        final_chunk = text[start:actual_end].strip()

        if len(final_chunk) > min_chunk_size:
            chunks.append({"text": final_chunk, "page": page_num})
            start = actual_end - overlap_size
        else:
            start = actual_end

    return chunks

def generate_advanced_chunks(page_content,page_num):
    search_chunks = generate_chunks_recursive(page_content,page_num,CHUNK_SIZE,OVERLAP_SIZE)

    for chunk in search_chunks:
        chunk["text"] = f"[Page {page_num}] {chunk['text']}"
        chunk["full_context"] = page_content

    return search_chunks

def generate_embeddings_batch(texts):
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        try:
            response = ollama.embed(model=EMBED_MODEL, input=batch_texts)
        except Exception as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc
        all_embeddings.extend(response["embeddings"])
    return all_embeddings

def search_numpy(query, vector_db, text_metadata):
    query = query.strip()
    if not query:
        return []

    response = ollama.embed(model=EMBED_MODEL, input=query)
    query_embedding = response["embeddings"][0]
    query_vector = np.array(query_embedding, dtype=np.float32)

    start_time = time.perf_counter()
    similarities = np.dot(vector_db, query_vector)
    top_indices = np.argsort(similarities)[-TOP_K:][::-1]

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Total time with numpy: {execution_time}")
    return [text_metadata[i] for i in top_indices]

def search_faiss(query, index, text_metadata):
    query = query.strip()
    if not query:
        return []

    response = ollama.embed(model=EMBED_MODEL, input=query)
    query_embedding = response["embeddings"][0]
    query_vector = np.array(query_embedding, dtype=np.float32)

    start_time = time.perf_counter()
    distances,indices = index.search(query_vector.reshape(1,-1), k=TOP_K)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Total time with FAISS: {execution_time}")
    return [text_metadata[i] for i in indices[0]]

def generate_answer(query, results, chat_history,threads,temp):
    options = {
        "num_thread": threads,
        "temperature": temp
        # "num_ctx": 8192  # Limits context window to keep it fast
    }

    raw_context_tokens = sum(estimate_tokens(res['full_context']) for res in results)
    context_parts = []
    for res in results:
        compressed_text = compress_context(query, res['full_context'])
        context_parts.append(
            f"--- START OF PAGE {res['page']} ---\n{compressed_text}\n--- END OF PAGE {res['page']} ---")

    compressed_context_text = "\n\n".join(context_parts).strip()
    compressed_tokens = estimate_tokens(compressed_context_text)

    stats = {
        "raw_tokens": raw_context_tokens,
        "compressed_tokens": compressed_tokens,
        "saved_tokens": raw_context_tokens - compressed_tokens
    }

    history_text = "\n".join([f"{history['role']}: {history['content']}" for history in chat_history[-5:]])

    context_text = "\n\n".join(context_parts).strip()
    prompt = f"""
You are a professional research assistant. Use the provided context to answer the question accurately.

Instructions:
1. Every time you state a fact from the context, cite the page number immediately after the sentence in brackets, like this: [Page X].
2. If the answer isn't in the context, clearly state that you don't know.
3. Keep your response structured and easy to read.

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
        model=THINKING_MODEL,
        prompt=prompt,
        stream=STREAM,
        keep_alive=KEEP_ALIVE,
        options=options,
    ),stats
    # response = ollama.generate(model=THINKING_MODEL, prompt=prompt, keep_alive=KEEP_ALIVE)
    # return response["response"],stats

def chat_pdf(index, text_metadata):
    while True:
        user_query = input("You - ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            break

        # results = search(user_query, vector_db, text_metadata)

        # results = search_faiss(user_query, index, text_metadata)
        results = search_with_rerank(user_query, index, text_metadata)
        # results1 = search_numpy(user_query, vector_db, text_metadata)
        if not results:
            print("No relevant content found for that query.")
            continue

        context_llm = [res["text"] for res in results]
        print("--- Response ---")
        if STREAM:
            for chunk in generate_answer(user_query, context_llm):
                print(chunk['response'], end='', flush=True)
            print("\n")  # New line after the full response is finished
        else:
            response = generate_answer(user_query, context_llm)
            print(f"{response}\n")

        pages = sorted({res["page"] for res in results})
        print(f"Sources: [Page {', '.join(map(str, pages))}]\n")

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

    print(f"Vector Magnitude: {norms}")

    if all_normalized:
        print("The embedding is L2 normalized.")
    else:
        print("The embedding is NOT normalized.")

def check_ollama_status():
    try:
        response = ollama.list()
        # downloaded_models=[model.get("model") for model in response.get("models",[])]
        downloaded_models = [m.model for m in response.models]
        is_model_missing = [model for model in [EMBED_MODEL,THINKING_MODEL] if model not in downloaded_models]

        if not is_model_missing:
            return True
        else:
            return False
    except Exception as exc:
        return False

def save_vector_db(index,metadata,current_hash):
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
    faiss.write_index(index, os.path.join(DB_FOLDER, "index.faiss"))
    data = {"metadata": metadata,"pdf_hash": current_hash}
    with open(os.path.join(DB_FOLDER, "metadata.pkl"), "wb") as f:
        pickle.dump(data, f)
    print("Saved vector database and PDF Hash value.")

def load_vector_db():
    if os.path.exists(DB_FOLDER):
        index = faiss.read_index(os.path.join(DB_FOLDER, "index.faiss"))
        with open(f"{DB_FOLDER}/metadata.pkl", "rb") as f:
            data = pickle.load(f)
        print("Database loaded successfully.")
        return index, data.get("metadata"), data.get("pdf_hash")
    return None, None, None

def calculate_pdf_hash(pdf_file):
    sha256_hash = hashlib.sha256()
    with open (pdf_file, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

@st.cache_resource
def get_ranker():
    return Ranker(model_name=RERANK_MODEL,cache_dir=CACHE_DIR)

def search_with_rerank(query,index,text_metadata):
    ranker = get_ranker()
    query = query.strip()
    if not query:
        return []

    page_match = re.search(r"page\s+(\d+)", query.lower())
    target_page = int(page_match.group(1)) if page_match else None
    response = ollama.embed(model=EMBED_MODEL, input=query)

    query_embedding = response["embeddings"][0]
    query_vector = np.array(query_embedding, dtype=np.float32)
    start_time = time.perf_counter()
    distances,indices = index.search(query_vector.reshape(1,-1), k=10)
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    candidates = [text_metadata[i] for i in indices[0]]

    if target_page:
        page_specific_chunks = [_text for _text in text_metadata if _text["page"] == target_page]
        candidates = page_specific_chunks[:3] + candidates

    rerank_items = [
        {
            "id": i,
            "text": candidate["text"],
            "page": candidate["page"],
            "full_context": candidate.get("full_context", "")  # Add this line
        }
        for i, candidate in enumerate(candidates)
    ]
    rerank_request = RerankRequest(query=query,passages=rerank_items)
    results = ranker.rerank(rerank_request)
    print(f"Total time with FAISS: {execution_time}")
    print("\n--- Re-ranker Scores ---")
    for i, res in enumerate(results[:TOP_K]):
        # FlashRank provides a 'score' key
        print(f"Rank {i + 1}: Score {res['score']:.4f} (Page {res['page']})")
    print("------------------------\n")
    return results[:TOP_K]

def compress_context(query, full_text):
    # Split into sentences
    sentences = re.split(r'(?<=[.!?]) +', full_text)
    query_words = set(query.lower().split())

    scored_sentences = []
    for s in sentences:
        # Score sentence based on keyword overlap with query
        score = sum(1 for word in s.lower().split() if word in query_words)
        scored_sentences.append((score, s))

    # Sort by score and take the top sentences, then re-sort by original order
    top_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:MAX_SENTENCES]
    # Re-sort to maintain document flow
    compressed = " ".join([s for _, s in top_sentences])

    return compressed if compressed else full_text[:1000]  # Fallback to snippet

def estimate_tokens(text):
    # Standard approximation: 1 token ≈ 4 characters or 0.75 words
    return len(text) // 4

def build_pipeline(pdf_file):
    if not check_ollama_status():
        print("Please start Ollama or run 'ollama pull <model_name>' for missing models.")
        st.warning("Please start Ollama or run 'ollama pull <model_name>' for missing models.")
        sys.exit(1)

    current_hash = calculate_pdf_hash(pdf_file)
    index, metadata,stored_hash = load_vector_db()

    if index and metadata and current_hash == stored_hash and not OVERRIDE_DB:
        st.info("Existing database found for this file. Loading...")
        return index, metadata

    st.info("New PDF detected. Processing...")
    texts = readpdf(pdf_file)
    if not texts:
        print("No text found in PDF.")
        st.warning("No text found in PDF.")
        sys.exit(1)

    metadata = []
    for page_num, page_content in texts:
        metadata.extend(generate_advanced_chunks(page_content, page_num))

    if not metadata:
        print("No content chunks were created from the PDF.")
        st.warning("No content chunks were created from the PDF.")
        sys.exit(1)

    text_data = [item["text"] for item in metadata]
    vectors = generate_embeddings_batch(text_data)
    verify_normalized_embedding(vectors)

    vector_np = np.array(vectors).astype('float32')

    dimension = vector_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vector_np)

    save_vector_db(index, metadata, current_hash)

    return index, metadata


# if __name__ == '__main__':
#     print('Welcome to Chat with PDF.')
#     index,metadata=build_pipeline(PDF_PATH)
#     chat_pdf(index, metadata)

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
            with st.spinner(text="Analyzing document...",show_time=True):
                idx,metadata=build_pipeline(tmp_path)
                st.session_state.index = idx
                st.session_state.metadata = metadata
                st.success(f"Ready to Chat!")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.subheader("⚙️ Performance Tuning")
        user_threads = st.slider("CPU Threads", 1, psutil.cpu_count(), get_safe_threads())
        use_flash = st.toggle("Flash Attention", value=True)
        temperature = st.slider("Creativity (Temp)", 0.0, 1.0, 0.1)

st.title("Chat with your PDF")

if "messages" not in st.session_state:
    st.session_state.messages = []

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
            response_placeholder=st.empty()
            full_response = ""

            results = search_with_rerank(prompt, st.session_state.index, st.session_state.metadata)

            start_gen = time.perf_counter()

            stream_response,stats = generate_answer(prompt, results, st.session_state.messages,user_threads,temperature)

            for chunk in stream_response:
                full_response += chunk['response']
                response_placeholder.markdown(full_response + "▌")

            end_gen = time.perf_counter()
            gen_time = end_gen - start_gen

            col1, col2, col3 = st.columns(3)
            col1.metric("Time Taken", f"{gen_time:.2f}s")
            col2.metric("Tokens Used", f"{stats['compressed_tokens']}")
            col3.metric("Tokens Saved", f"{stats['saved_tokens']}", delta_color="normal")

            # for chunk in generate_answer(prompt, results, st.session_state.messages):
            #     full_response += chunk['response']
            #     response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            pages = sorted({res["page"] for res in results})
            st.caption(f"Sources: Pages {', '.join(map(str, pages))}")

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # vector_db = np.array(vectors, dtype=np.float32)
    # text_metadata = all_metadata
    # chat_pdf(index,vector_db, text_metadata)

    # user_query = "What is the main topic of the PDF?"
    # user_query = "What does this mention about Heuristic based search?"
    # user_query="What is explanation facility? Explain with any examples provided by PDF."
    # user_query="Is anything mentioned as Knowledge representation and inference, and any examples provided for it?"
    # user_query="What is the main topic of the PDF? What are the subtopics covered in it?"
    # user_query="Explain in detail about Heuristic based search" # failing question
    # results = search(user_query, vector_db, text_metadata)
    #
    # response = generate_answer(user_query, chunks)
    # print(f"--- Query ---\n{user_query}\n")
    # print(f"--- Response ---\n{response}\n")