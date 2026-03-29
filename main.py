import pdfplumber
import ollama
import numpy as np

PDF_PATH = "AI Module.pdf"
CHUNK_SIZE = 1024
OVERLAP_SIZE = 200
EMBED_MODEL = "nomic-embed-text"
THINKING_MODEL = "llama3.1:latest "
BATCH_SIZE=32
TOP_K=3

def readpdf():
    print("Reading PDF...")
    all_texts=[];
    with pdfplumber.open(PDF_PATH) as pdf:
        for i,page in enumerate(pdf.pages):
            text = page.extract_text()
            all_texts.append(text)
    return all_texts

def generate_chunks(text):
    chunks =[]
    i=0
    while i<len(text):
        end = min (i+CHUNK_SIZE, len(text))
        chunk = text[i:end]

        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space != -1:
                end = i+ last_space
                chunk = chunk[i:end]

        chunks.append(chunk)
        i = end - OVERLAP_SIZE

        if i >= len(text) or (end == len(text)):
            break

    return chunks
    # print all chunks
    # for chunk in chunks:
    #     print(chunk)

    # for i in range (0, len(text), CHUNK_SIZE-OVERLAP_SIZE):
    #     chunk = text[i:i+CHUNK_SIZE]
    #     print (chunk)

def generate_embeddings_batch(chunks):
    all_embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks= chunks[i:i+BATCH_SIZE]
        response = ollama.embed(model=EMBED_MODEL, input=batch_chunks)
        all_embeddings.extend(response["embeddings"])
    return all_embeddings

def search(query, vector_db, text_metadata):
    query_embedding = ollama.embed(model=EMBED_MODEL, input=query)["embeddings"][0]
    query_vector=np.array(query_embedding)
    similarities = np.dot(vector_db, query_vector)
    top_indices = np.argsort(similarities) [-TOP_K:][::-1]
    return [text_metadata[i] for i in top_indices]

def generate_answer(query, chunks):
    context_chunks="".join(chunks)

    prompt = f"""
    You are a helpful assistant. Use the following pieces of retrieved context 
    to answer the user's question. If you don't know the answer based on the 
    context, just say that you don't know.

    Context:
    {context_chunks}

    Question: 
    {query}

    Answer:
    """

    response = ollama.generate(model=THINKING_MODEL, prompt=prompt)
    # print(response)
    # return response["answers"][0]["text"]
    return response["response"]

def chat_pdf(vector_db,text_metadata):

    while(True):
        user_query = input("You - ")
        if user_query == 'exit':
            break
        results = search(user_query, vector_db, text_metadata)
        response = generate_answer(user_query, results)
        print(f"--- Response ---\n{response}\n")

def print_all(text): # print all texts
    chunks = 1024
    for i in range(0, len(final_text), chunks):
        print(final_text[i:i + chunks])

if __name__ == '__main__':
    print('Welcome to Chat with PDF.')
    texts = readpdf()
    final_text = "".join(texts)
    chunks = generate_chunks(final_text)
    vectors = generate_embeddings_batch(chunks)

    vector_db = np.array(vectors)
    text_metadata = chunks
    chat_pdf(vector_db,text_metadata)

    # user_query = "What is the main topic of the PDF?"
    # user_query = "What does this mention about Heuristic based search?"
    # user_query="What is explanation facility? Explain with any examples provided by PDF."
    # user_query="Is anything mentioned as Knowledge representation and inference, and any examples provided for it?"
    # results = search(user_query, vector_db, text_metadata)
    #
    # response = generate_answer(user_query, chunks)
    # print(f"--- Query ---\n{user_query}\n")
    # print(f"--- Response ---\n{response}\n")

    # for r in results:
    #     print(f"--- Found Chunk ---\n{r}\n")
    # print_all(final_text)
