from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_faiss_index(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    return db

def search_similar_documents(query, faiss_index, k=3): # you can change k = for more or less similarity searches
    return faiss_index.similarity_search(query, k=k)

