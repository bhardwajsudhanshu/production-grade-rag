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
os.environ["LANGSMITH_PROJECT"] = "multi-agent-research-system"

def demo_basic_tracing():
    """ basic langsimth tracing"""
    llm = init_chat_model(
            model="gemini-3.6-flash", 
            temperature=0.7,
            model_provider="google_genai"
    )