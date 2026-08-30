from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
def demo_contextual_compression():
    """contextual compression extracts only releveant parts"""

    print("=" * 60)
    print("CONTEXTUAL COMPRESSION RETRIEVER")
    print("extracts only query-relevant content from documents")
    print("=" * 60)

    vectorstore = create_base_vectorstore()
    llm = ChatGoogleGenerativeAI(
        model='gemini-3.6-flash',
        temperature=0
    )

    #create compressor
    compressor = LLMChainExtractor.from_llm(llm)

    #wrap retriever with compression
    compression_retriever = ContextualCompressionRetriever(
        base_compressor = compressor,
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    )

    query = "what framewroks exist for building llm applications?"

    print(f"\nQuery: {query}")

    #without compression
    base_docs = vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(query)
    print(f"\n--- WITHOUT Compression (full chunks) ---")
    for doc in base_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Cntent: {doc.page_content[:150]}...\n")


    # with compression
    compressed_docs = compression_retriever.invoke(query)
    print(f"\n--- WITH Compression (relevant only) ---")
    for doc in compressed_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Cntent: {doc.page_content}\n")

