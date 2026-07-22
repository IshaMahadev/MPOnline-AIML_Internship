import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from rag_utils import process_documents, get_conversational_chain
import ui

# Load environment variables
load_dotenv()

# --- Page Configuration & Styling ---
ui.setup_page()
ui.render_header()

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Sidebar Configuration ---
sidebar_config = ui.render_sidebar()

# Determine the active API key (User input overrides the hidden secret)
active_google_api_key = sidebar_config["api_key_input"] if sidebar_config["api_key_input"] else os.environ.get("GOOGLE_API_KEY", "")

# Reset chat if API key or Vector DB changes (we track previous values to know when to reset)
if "prev_api_key" not in st.session_state or st.session_state.prev_api_key != active_google_api_key:
    st.session_state.prev_api_key = active_google_api_key
    st.session_state.vector_store = None
    st.session_state.chat_history = []
    st.session_state.messages = []

# --- Process Document ---
if sidebar_config["process_btn"]:
    if not sidebar_config["uploaded_file"]:
        st.sidebar.error("Please upload a PDF file first.")
    else:
        with st.spinner("Processing document..."):
            try:
                active_pinecone_api_key = sidebar_config["pinecone_api_key"] if sidebar_config["pinecone_api_key"] else os.environ.get("PINECONE_API_KEY", "")
                active_pinecone_index = sidebar_config["pinecone_index"] if sidebar_config["pinecone_index"] else os.environ.get("PINECONE_INDEX_NAME", "")
                
                vector_store, num_chunks = process_documents(
                    uploaded_file=sidebar_config["uploaded_file"],
                    vector_db_type=sidebar_config["vector_db"],
                    pinecone_api_key=active_pinecone_api_key,
                    pinecone_index=active_pinecone_index
                )
                st.session_state.vector_store = vector_store
                st.session_state.chat_history = []
                st.session_state.messages = []
                st.sidebar.success(f"Successfully processed {num_chunks} chunks into {sidebar_config['vector_db']}!")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")

# --- Chat Interface ---
# Display chat messages from history
ui.render_chat_messages(st.session_state.messages)

# Handle user input
prompt = ui.get_chat_input()
if prompt:
    if not active_google_api_key:
        st.error("Please provide a Google Gemini API Key in the sidebar or via secrets.")
        st.stop()
        
    if st.session_state.vector_store is None:
        st.error("Please upload and process a document first.")
    else:
        # Show user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if not active_google_api_key:
            st.error("⚠️ No Google Gemini API Key found. Please enter it in the sidebar or configure Streamlit Secrets.")
            st.stop()

        try:
            # Initialize the LLM here to avoid import caching issues
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                google_api_key=active_google_api_key,
                model="gemini-3-flash-preview",
                temperature=0.3
            )
            
            rag_chain = get_conversational_chain(
                vector_store=st.session_state.vector_store,
                llm=llm
            )
            
            # Use ui module to render the assistant thinking and generating response
            answer = ui.render_assistant_thinking(
                prompt=prompt, 
                llm=llm, 
                rag_chain=rag_chain, 
                chat_history=st.session_state.chat_history
            )
            
            # Update Langchain chat history memory
            st.session_state.chat_history.extend([
                HumanMessage(content=prompt),
                AIMessage(content=answer)
            ])
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
