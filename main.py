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

PDF_PATH = os.environ.get("PDF_PATH", "AI Module.pdf")
CHUNK_SIZE = 1024
OVERLAP_SIZE = 200
EMBED_MODEL = "nomic-embed-text:latest"
THINKING_MODEL = "llama3.1:latest"
BATCH_SIZE = 32
TOP_K = 3
STREAM = True
KEEP_ALIVE = '1h'
DB_FOLDER = "db"
OVERRIDE_DB = False

def readpdf():
    print(f"Reading PDF: {PDF_PATH}")
    all_texts = []

    try:
        with pdfplumber.open(PDF_PATH) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                all_texts.append((i + 1, text))
    except FileNotFoundError:
        print(f"PDF not found: {PDF_PATH}")
        sys.exit(1)
    except Exception as exc:
        print(f"Failed to read PDF: {exc}")
        sys.exit(1)

    return all_texts

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

def generate_answer(query, chunks):
    context_chunks = "\n\n".join(chunks).strip()
    prompt = f"""
You are a helpful assistant. Use the following pieces of retrieved context to answer the user's question. If you don't know the answer based on the context, just say that you don't know.

Context:
{context_chunks}

Question:
{query}

Answer:
"""
    if STREAM:
        return ollama.generate(
            model=THINKING_MODEL,
            prompt=prompt,
            stream=STREAM,
            keep_alive=KEEP_ALIVE
        )
    response = ollama.generate(model=THINKING_MODEL, prompt=prompt, keep_alive=KEEP_ALIVE)
    return response["response"]

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

# verify if ollama is up and running
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

def calculate_pdf_hash():
    sha256_hash = hashlib.sha256()
    with open (PDF_PATH, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2",cache_dir="/tmp")
def search_with_rerank(query,index,text_metadata):
    qquery = query.strip()
    if not query:
        return []

    response = ollama.embed(model=EMBED_MODEL, input=query)

    query_embedding = response["embeddings"][0]
    query_vector = np.array(query_embedding, dtype=np.float32)
    start_time = time.perf_counter()
    distances,indices = index.search(query_vector.reshape(1,-1), k=10)
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    candidates = [text_metadata[i] for i in indices[0]]
    rerank_items = [
        {"id":i, "text":candidate["text"],"page":candidate["page"]}
        for i,candidate in enumerate(candidates)]
    rerank_request = RerankRequest(query=query,passages=rerank_items)
    results = ranker.rerank(rerank_request)
    print(f"Total time with FAISS: {execution_time}")
    print("\n--- Re-ranker Scores ---")
    for i, res in enumerate(results[:TOP_K]):
        # FlashRank provides a 'score' key
        print(f"Rank {i + 1}: Score {res['score']:.4f} (Page {res['page']})")
    print("------------------------\n")
    return results[:TOP_K]

if __name__ == '__main__':
    print('Welcome to Chat with PDF.')
    if not check_ollama_status():
        print("Please start Ollama or run 'ollama pull <model_name>' for missing models.")
        sys.exit(1)

    current_hash = calculate_pdf_hash()

    index, metadata,stored_hash = load_vector_db()
    if not index or current_hash != stored_hash or OVERRIDE_DB:
        texts = readpdf()
        if not texts:
            print("No text found in PDF.")
            sys.exit(1)

        metadata = []
        for page_num, page_content in texts:
            metadata.extend(generate_chunks(page_content, page_num=page_num))

        if not metadata:
            print("No content chunks were created from the PDF.")
            sys.exit(1)

        text_data = [item["text"] for item in metadata]
        vectors = generate_embeddings_batch(text_data)
        verify_normalized_embedding(vectors)

        vector_np = np.array(vectors).astype('float32')

        dimension = vector_np.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vector_np)

        save_vector_db(index, metadata,current_hash)

    chat_pdf(index, metadata)

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