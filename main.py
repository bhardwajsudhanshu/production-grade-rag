from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()  # Load environment variables from .env file

core_version = version("langchain-core")
lg_version = version("langgraph")
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq



print(f"Langchain Core Version: {core_version}")
print(f"LangGraph Version: {lg_version}")






def main():
    llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0.7)
    response = llm.invoke("say setup complete in one word!")
    print(f"response form Google genai: {response}")

    llm_groq = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
    response_groq = llm_groq.invoke("say setup complete in one word!")
    print(f"response form Groq: {response_groq}")

    print("setup complete!")


if __name__ == "__main__":
    main()
