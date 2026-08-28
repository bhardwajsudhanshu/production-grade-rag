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

