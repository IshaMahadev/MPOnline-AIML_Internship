import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import tempfile

load_dotenv()

def process_documents(uploaded_file, vector_db_type, pinecone_api_key=None, pinecone_index=None):
    """Loads PDF, splits text, creates embeddings, and initializes the selected vector store."""
    # 1. Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # 2. Load and Split Document
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    
    # 3. Embeddings (1024 dims to match Pinecone)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    
    # 4. Vector Store Setup
    vector_store = None
    if vector_db_type == "Chroma (Local)":
        vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
    elif vector_db_type == "Pinecone (Cloud)":
        if not pinecone_api_key or not pinecone_index:
            raise ValueError("Pinecone API Key and Index Name are required.")
        os.environ["PINECONE_API_KEY"] = pinecone_api_key
        pc = Pinecone(api_key=pinecone_api_key)
        vector_store = PineconeVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            index_name=pinecone_index
        )
    
    # Cleanup temp file
    os.remove(tmp_file_path)
    return vector_store, len(splits)

def get_conversational_chain(vector_store, llm):
    """Creates a conversational RAG chain with chat history awareness using LCEL."""

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # --- Prompt 1: Rephrase user question considering chat history ---
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "reformulate a standalone question that can be understood without the history. "
         "Do NOT answer it — just rephrase if needed, otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    # --- Prompt 2: Answer based on retrieved context ---
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant for question-answering tasks. "
         "Use the retrieved context below to answer the question. "
         "If you don't know the answer, say so. Keep answers concise.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_standalone_question(data):
        """Rephrase the question using chat history."""
        if not data.get("chat_history"):
            return data["question"]
        chain = contextualize_q_prompt | llm | StrOutputParser()
        return chain.invoke(data)

    # Full RAG chain using LCEL
    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=RunnableLambda(get_standalone_question)
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_docs(retriever.invoke(x["standalone_question"])))
        )
        | RunnablePassthrough.assign(
            answer=qa_prompt | llm | StrOutputParser()
        )
    )

    return rag_chain
