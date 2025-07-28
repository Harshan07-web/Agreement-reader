from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

def rag_chain(vectorstore):
    retriever = RunnableLambda(lambda x: vectorstore.similarity_search(x["question"]))
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    prompt = ChatPromptTemplate.from_template("""
                                "You are a legal assistant ,answer the questions based" \
    "on the infromation from the document onyl. If you cannot answer,then say 'not found in agreement'"
    context : {context}

    question:{question}
    """)

    rag_chain = (
        {"context": retriever, "question": RunnableLambda(lambda x: x["question"])}
        | prompt | model | StrOutputParser()
    )

    return rag_chain

    