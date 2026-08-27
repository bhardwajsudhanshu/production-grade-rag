from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

documents = [
    Document(
        page_content='product SKU-7742X is our flagship router. It supports'
                     'gigabit speeds and has advanced qos features',
        metadata={'type': 'product'}
    ),
    Document(
        page_content='for network connectivity issues, first check the'
                     'ethernet cable and router status lights',
        metadata={'type': 'troubleshooting'}
    ),
    Document(
        page_content='Error code E_CONN_REFUSED indicates the server'
                     'rejected the connection. Check firewall settings.',
        metadata={'type': 'error'}
    ),
    Document(
        page_content='The authentication process requires a valid credentials.'
                     'use 0Auth2 for secure API access',
        metadata={'type': 'auth'}
    ),
    Document(
        page_content='Router configuration guide: Access the admin panel'
                     'at 192.168.1.1 to modeify settings',
        metadata={'type': 'config'}
    ),
    Document(
            page_content='WCAG 2.1 compliance requires all images to have'
                         'alt text and sufficient color contrast',
            metadata={'type': 'compliance'}
    ),
]

print(f'Loaded {len(documents)} documents')

vectorstore = Chroma.from_documents(
    documents,
    embeddings,
    collection_name='hybrid_test'

)
# create vector retriever
vector_retriever = vectorstore.as_retriever(
    search_kwargs={'k':3} # returnh top 3
)

print('vector retriever ready')

# BM25 works on the raw text
bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=3 # return top 3
)
print('BM25 retriever ready')

# combine with ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers = [bm25_retriever, vector_retriever],
    weights = [0.5, 0.5] # equal weight to both
)
print('hybrid retrieever ready')

def test_query(query, name, retriever):
    '''test a query and dhow results'''
    results = retriever.invoke(query)
    print(f'\\n{name} - Query: \"{query}\"')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + '...'
        print(f' {i+1}. {preview}')
    return results

# test queries designed to challenge vector search
test_queries = [
    'SKU-7742X specifications', # extract product code
    'E_CONN_REFUSED error', # error code
    'How do I authenticate', # semantic question
    'WCAG compliance', # acronym
    'router configuration' # genearal semantic
]

for query in test_queries:
    print('=' * 60)

    # vector only
    vector_results = test_query(query, 'VECTOR', vector_retriever)

    #BM25 only
    bm25_results = test_query(query, 'BM25', bm25_retriever)

    hybrid_results = test_query(query, 'HYBRID', ensemble_retriever)
