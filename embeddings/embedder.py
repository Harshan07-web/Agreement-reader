import os 
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_load_vectorstore(docs, index_path="faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="AIzaSyBnXqsczmdQ5uRwpOabS-m4vNJ9ue_lySU"
    )
    return FAISS.from_documents(docs, embeddings)
