import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from embeddings.embedder import create_load_vectorstore
from loaders.pdf_loader import load_sample_file
from retriever.rag_pipeline import rag_chain


def main():
    file_path = "D:\\Agreement Reader\\data\\Law_Insider_Assistant_Document.pdf"
    print("loading data to loder")
    documents = load_sample_file(file_path)

    print("creating the vectorstore")
    vectorstore = create_load_vectorstore(documents)

    print("send to pipeline")
    chain = rag_chain(vectorstore)

    print("agreement is ready ,ask a question\n")
    while True:
        query = input("Enter your question, send 'exit' to exit: ")
        if query.lower() == 'exit':
            break

        answer = chain.invoke({"question":query})
        print("Answer:\n",answer)
if __name__ == "__main__":
    main()





