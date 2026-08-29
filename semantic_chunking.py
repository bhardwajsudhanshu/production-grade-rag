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

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=['\n\n','\n','. ',' ']
)

recursive_chunks = recursive_splitter.split_text(document)


print(f"Recursive chunks: {len(recursive_chunks)}")
for i, chunk in enumerate(recursive_chunks):
    print(f"\\n--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk[:100] + "..." if len(chunk) > 100 else chunk)

semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=90 #split at 90th percentile dissimilarity
)
semantic_chunks = semantic_chunker.split_text(document)

print(f"Semantic Chunks: {len(semantic_chunks)}")
for i, chunk in enumerate(semantic_chunks):
    print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk[:100] + "..." if len(chunk) > 100 else chunk)

# Filter out empty or whitespace-only chunks to prevent Gemini API errors
filtered_recursive_chunks = [c for c in recursive_chunks if c.strip()]
filtered_semantic_chunks = [c for c in semantic_chunks if c.strip()]


# create two vectors 1 for each chunking method
recursive_vectorstore = Chroma.from_texts(
    filtered_recursive_chunks,
    embeddings,
    collection_name='recursive_chunks'
)

semantic_vectorstore = Chroma.from_texts(
    filtered_semantic_chunks,
    embeddings,
    collection_name='semantic_chunks'
)

# test queries
test_queries = [
    'How do i authenticate with 0auth2?',
    'What happens when i hit the rate limit?',
    'How are Webhoks secured?',
    'What formats are errors returned in?'
]

def test_retrieval(query, vectorstore, name):
    results = vectorstore.similarity_search(query, k=1)
    print(f'\\n{name} - Query: \"{query}\"')
    print(f'Retrieved: {results[0].page_content[:150]}...')
    return results[0].page_content

print(f"\n{'='*60}")
print(" RETRIEVAL TESTS")
print(f"{'='*60}")

for query in test_queries:
    print('='*60)
    recursive_result = test_retrieval(query, recursive_vectorstore, 'RECURSIVE')
    semantic_result = test_retrieval(query, semantic_vectorstore, 'SEMANTIC')