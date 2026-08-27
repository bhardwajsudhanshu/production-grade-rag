import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langsmith import traceable
from langsmith.run_trees import RunTree
from dotenv import load_dotenv

load_dotenv()

# Enable tracing
os.environ["LANGSMITH_TRACING"] = "true"
#os.environ["LANGSMITH_PROJECT"] = "multi-agent-research-system"

@traceable(name="basic_chaining")
def demo_basic_tracing():
    """ basic langsimth tracing"""
    llm = init_chat_model(
            model="gemini-3.6-flash", 
            temperature=0,
            model_provider="google_genai"
    )
    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in one sentence."
    )

    chain = prompt | llm | StrOutputParser()

    print("Basic Training demo:\n")
    print("Running chain with Langsmith tracing enabled...")

    result = chain.invoke({"topic": "machine learning"})

    print(f"result: {result}")
    print("\nCheck Langsmith dashboard for trace details.")

@traceable(name="named_runs_demo", tags=["production", "summarization"])
def demo_named_runs():
    """Name your runs for easier identification."""

    llm = init_chat_model(
            model="gemini-3.6-flash", 
            temperature=0,
            model_provider="google_genai"
    )

    prompt = ChatPromptTemplate.from_template("Summarize: {text}")

    chain = prompt | llm | StrOutputParser()

    print("\nNamed Runs Demo:\n")

    result = chain.invoke(
        {"text": "LangSmith provides observability for LLM applications."}
    )

    print(f"Result: {result}")
    print("Run tagged with 'production', 'summarization'")

@traceable(name="trace_with_metadata_demo", tags=["metadata", "filtering"])
def demo_trace_with_metadata(user_id: str, request_type: str):
    """Add metadata to traces for filtering."""

    llm = init_chat_model(
            model="gemini-3.6-flash", 
            temperature=0,
            model_provider="google_genai"
    )

    # Metadata is automatically captured
    result = llm.invoke(f"Hello from user {user_id}")

    return result.content

if __name__ == "__main__":
    demo_basic_tracing()
    demo_named_runs()
    demo_trace_with_metadata(user_id="user_123", request_type="greeting")
