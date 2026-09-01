import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from the .env file
load_dotenv()

def generate_text():
    # The client automatically picks up GOOGLE_APPLICATION_CREDENTIALS
    client = genai.Client()
    
    # Using the latest default multimodal flash model
    model_id = 'gemini-3.6-flash' 
    prompt = 'Explain the difference between synchronous and asynchronous programming in two sentences.'

    print("Sending request to Vertex AI...")
    
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
    )
    
    print("\n--- Gemini Response ---")
    print(response.text)

if __name__ == "__main__":
    # Quick sanity check to ensure environment variable is loaded
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
    else:
        generate_text()
