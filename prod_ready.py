from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

document = '''
# Authentication guide
# 
## 0Auth2 Authentication
To authenticate with our API, you need 0Auth2 credentials.
first, obtain a client_id and client_secret from the developer portal.
make a POST request to /oauth/token with grant_type=client_crdentials
The response contains an access_token valid for 3600 seconds.
Include this token in the Authorization header as 'Bearer <token>'.

## Rate limiting
our API implements rate limiting using a token bucket algorithm.
Free tier: 100 requests per minute.
Pro tier: 1000 requests per minute.
## Error Handling
All errors return a standard JSON format.
The 'code' field contains a machine-readable error code.
The 'message' field contains a human-readable description.
Common errors: AUTH_FAILED, RATE_LIMITED, INVALID_REQUEST.
Always check the HTTP status code first, then parse the error body.

## Webhooks
configure webhooks in your dashboard settings.
We support HTTP and HTTPS endpoints.
Webhook payloads are signed with HMAC-SHA256.
Verify signatures using your webhook secret.
Failed deliveries are retired with exponential backoff.
'''
def smart_chunker(
    text: str,
    use_semantic: bool = True,
    fallback_chunk_size:int = 500
) -> list[str]:
    '''
    Production chunking with semantic as primary, recursive as fallback.
    '''
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    if use_semantic:
        try:
            chunker = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90,
            )
            chunks = chunker.split_text(text)
            
            # validate chunks are not too large
            max_chunk_size = 2000
            if any(len(c) > max_chunk_size for c in chunks):
                # fallback to recursive for oversized chunks
                return  _recursive_fallback(text,fallback_chunk_size)
                
            return chunks

        except Exception as e:
            print(f"semantic chunking failed: {e}")
            return _recursive_fallback(text,fallback_chunk_size)
            
    return _recursive_fallback(text,fallback_chunk_size)
    
def _recursive_fallback(text:str,chunk_size:int)->list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
    )
    return text_splitter.split_text(text)

#usage
chunks = smart_chunker(document, use_semantic=True)
print(f'Created {len(chunks)} semantic chunks')

