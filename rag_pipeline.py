import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Resolve relative credential path to absolute
creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if creds and not os.path.isabs(creds):
    base = os.path.dirname(os.path.abspath(__file__))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(base, creds)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "kaggle-submission2")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
API_KEY = os.environ.get("GOOGLE_API_KEY")

# Embeddings via Vertex AI (authenticated with service account)
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

# Chat via Gemini Developer API (Vertex AI generative models not enabled in this project)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=API_KEY,
    vertexai=False,
)


# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""


# now we will create knowledge base
def create_kb():
    """create a vector store from knowledge base."""

    # split the knowledge base into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE,
                    metadata={"source": "langchain_knwoledge_base.md"})

    chunks = splitter.split_documents([doc])

    # create a vector satore from chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )

    return vector_store



def demo_basic_rag():

    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2}) # kwargs 2 we just want 2 documents to be retrieved

    # RAG prompt template
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context:

{context}

Question: {question}

Answer:


Make sure to answer in a concise manner, 
and if you don't know the answer, just say "I don't know."""
    )

    # format retrieved docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # RAG chain
    rag_chain = (
        {"context":retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # test the rag chain
    questions = [
        "what is LangChain?",
        "What is LangGraph?",
        "who created LangChain?",

    ]

    print("Basic RAG Demo:\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

def demo_rag_with_sources():
    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # retrieve 3 documents

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based on the context below. Include which sources you used.

Context:
{context}

Question: {question}

Answer (include sources):"""
    )

    def format_docs_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'unknown')
            formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        |StrOutputParser()
    )

    print("RAG with Sources:\n")
    answer = rag_chain.invoke("What are the components of LangChain?")
    print(f"Q: What are the core components?\n")
    print(f"A: {answer}")
                                      
if __name__ == "__main__":
    #demo_basic_rag()
    demo_rag_with_sources()


