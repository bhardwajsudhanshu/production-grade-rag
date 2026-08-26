from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np
#from ollama import embeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def basic_embeddings():

    # single text
    text = "what is machine learning?"
    single_embedding = embeddings.embed_query(text)

    print(f"Vector Dimensions: {len(single_embedding)}")
    print(f"first 5 values: {single_embedding[:5]}")
    print(f"vector norm: {np.linalg.norm(single_embedding):.4f}")

def batch_embeddings():
    text = [
        "what is machine learning?",
        "what is deep learning?",
        "what is reinforcement learning?"
    ]
    batch_embeddings = embeddings.embed_documents(text)
    for i, emb in enumerate(batch_embeddings):
        print(f"Text {i+1} - vector Dimensions: {len(emb)}")
        print(f"Text {i+1} - first 5 values: {emb[:5]}")
        print(f"Text {i+1} - vector norm: {np.linalg.norm(emb):.4f}")

def similarity_search():
    docs = [
        "python is a programming language",
        "javascript is used for web development",
        "machine learning enables ai applications",
        "deep learning uses neural networks",
        "cats are popular pets",
    ]

    query = "what programming languages exist?"

    # embed documants and query
    doc_vector = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)

    # compute cosine similarity
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    # Convert similarity scores to a flat list of standard Python floats
    #scalar_similarities = [float(sim) for sim in similarities]

    # rank documents by similarity
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by similarity:")
    for doc, score in ranked_docs:
        print(f"  {score:.4f}: {doc}")



if __name__ == "__main__":
    #basic_embeddings()
    #batch_embeddings()
    similarity_search()


