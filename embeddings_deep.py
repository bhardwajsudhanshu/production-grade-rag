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


if __name__ == "__main__":
    #basic_embeddings()
    batch_embeddings()


