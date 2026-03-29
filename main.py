import os
import sys
import numpy as np
import ollama
import pdfplumber

PDF_PATH = os.environ.get("PDF_PATH", "AI Module.pdf")
CHUNK_SIZE = 1024
OVERLAP_SIZE = 200
EMBED_MODEL = "nomic-embed-text:latest"
THINKING_MODEL = "llama3.1:latest"
BATCH_SIZE = 32
TOP_K = 3

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

def search(query, vector_db, text_metadata):
    query = query.strip()
    if not query:
        return []

    response = ollama.embed(model=EMBED_MODEL, input=query)
    query_embedding = response["embeddings"][0]
    query_vector = np.array(query_embedding, dtype=np.float32)
    similarities = np.dot(vector_db, query_vector)
    top_indices = np.argsort(similarities)[-TOP_K:][::-1]
    return [text_metadata[i] for i in top_indices]

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

    response = ollama.generate(model=THINKING_MODEL, prompt=prompt)
    return response["response"]

def chat_pdf(vector_db, text_metadata):
    while True:
        user_query = input("You - ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            break

        results = search(user_query, vector_db, text_metadata)
        if not results:
            print("No relevant content found for that query.")
            continue

        context_llm = [res["text"] for res in results]
        response = generate_answer(user_query, context_llm)
        print(f"--- Response ---\n{response}\n")

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
        downloaded_models=[model.get("model") for model in response.get("models",[])]
        is_model_missing = [model for model in [EMBED_MODEL,THINKING_MODEL] if model not in downloaded_models]

        if not is_model_missing:
            return True
        else:
            return False
    except Exception as exc:
        return False

if __name__ == '__main__':
    print('Welcome to Chat with PDF.')
    if not check_ollama_status():
        print("Please start Ollama or run 'ollama pull <model_name>' for missing models.")
        sys.exit(1)
    texts = readpdf()
    if not texts:
        print("No text found in PDF.")
        sys.exit(1)

    all_metadata = []
    for page_num, page_content in texts:
        all_metadata.extend(generate_chunks(page_content, page_num=page_num))

    if not all_metadata:
        print("No content chunks were created from the PDF.")
        sys.exit(1)

    text_data = [item["text"] for item in all_metadata]
    vectors = generate_embeddings_batch(text_data)
    verify_normalized_embedding(vectors)  # Check the first embedding for normalization
    if not vectors:
        print("No embeddings were generated.")
        sys.exit(1)

    vector_db = np.array(vectors, dtype=np.float32)
    text_metadata = all_metadata
    chat_pdf(vector_db, text_metadata)

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