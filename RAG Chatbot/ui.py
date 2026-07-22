import streamlit as st
import os

def setup_page():
    """Sets up the Streamlit page configuration and custom CSS styling."""
    st.set_page_config(
        page_title="Knowledge Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for premium aesthetics
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* Global Font */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* App Background */
        .stApp {
            background-color: #0b0f19;
            color: #e2e8f0;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #1f2937;
        }

        /* Main Header Gradient */
        .main-header {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }

        /* Subheader */
        .sub-header {
            color: #94a3b8;
            font-size: 1.1rem;
            font-weight: 300;
            margin-bottom: 2.5rem;
            line-height: 1.6;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        }

        /* Inputs */
        .stTextInput>div>div>input {
            background-color: #1f2937;
            color: #f8fafc;
            border: 1px solid #334155;
            border-radius: 8px;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 1px #3b82f6;
        }

        /* Chat Input Container */
        .stChatFloatingInputContainer {
            padding-bottom: 30px;
            background-color: transparent;
        }
        
        /* File Uploader */
        [data-testid="stFileUploadDropzone"] {
            background-color: #1f2937;
            border: 1px dashed #475569;
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        [data-testid="stFileUploadDropzone"]:hover {
            border-color: #3b82f6;
            background-color: #1e293b;
        }

        /* Remove default Streamlit top padding */
        .block-container {
            padding-top: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Renders the main title and description of the application."""
    st.markdown('<h1 class="main-header">Knowledge Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask anything about your uploaded documents. I can remember our chat history for follow-up questions!</p>', unsafe_allow_html=True)

def render_sidebar():
    """Renders the sidebar configuration and file uploader."""
    with st.sidebar:
        st.header("Configuration")
        
        # API Key Input
        api_key_input = st.text_input(
            "Google Gemini API Key (Optional)",
            value="",
            type="password",
            help="Leave blank to use the default app key."
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Vector DB Selection
        vector_db = st.radio("Select Vector Database:", ("Chroma (Local)", "Pinecone (Cloud)"))
        
        # Pinecone specific inputs if selected
        pinecone_api_key = ""
        pinecone_index = ""
        if vector_db == "Pinecone (Cloud)":
            pinecone_api_key = st.text_input(
                "Pinecone API Key (Optional)", 
                type="password", 
                value="",
                help="Leave blank to use the default app key."
            )
            pinecone_index = st.text_input(
                "Pinecone Index Name (Optional)", 
                value="",
                help="Leave blank to use the default app index."
            )

        
        st.markdown("<br><hr style='border-color: #1f2937;'><br>", unsafe_allow_html=True)
        
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        
        st.markdown("<br>", unsafe_allow_html=True)
        process_btn = st.button("Process Document", use_container_width=True)
        
        return {
            "api_key_input": api_key_input,
            "vector_db": vector_db,
            "pinecone_api_key": pinecone_api_key,
            "pinecone_index": pinecone_index,
            "uploaded_file": uploaded_file,
            "process_btn": process_btn
        }

def render_chat_messages(messages):
    """Displays chat messages from the conversation history."""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def get_chat_input():
    """Gets the user input from the chat input box."""
    return st.chat_input("Ask a question about your document...")

def render_assistant_thinking(prompt, llm, rag_chain, chat_history):
    """Renders the assistant thinking process and generates the response."""
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({
                "question": prompt,
                "chat_history": chat_history
            })
            answer = response["answer"]
            st.markdown(answer)
            return answer
