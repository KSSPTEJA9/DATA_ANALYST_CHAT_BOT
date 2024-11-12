import streamlit as st
from streamlit_chat import message
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
import os
import hashlib

# Paths for the FAISS database and model
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_FOLDER_PATH = "data"
SAVED_FILE_PATH = os.path.join(DATA_FOLDER_PATH, "uploaded_file.csv")

# Ensure the data folder exists
os.makedirs(DATA_FOLDER_PATH, exist_ok=True)

# Initialize session states for chat history, database, qa chain, and file hash
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "db_initialized" not in st.session_state:
    st.session_state["db_initialized"] = False
if "last_file_hash" not in st.session_state:
    st.session_state["last_file_hash"] = None
if "qa" not in st.session_state:
    st.session_state["qa"] = None
if "user_input_submitted" not in st.session_state:
    st.session_state["user_input_submitted"] = False

# Function to hash the uploaded file for change detection
def calculate_file_hash(file):
    file_content = file.getvalue()
    return hashlib.md5(file_content).hexdigest()

# Streamlit UI components
st.title("Chat with Your CSV Data 📊")

# File upload component
uploaded_file = st.file_uploader("Upload a CSV file to chat with", type="csv")

# Process the uploaded file if it's new or different
if uploaded_file is not None:
    # Calculate hash to check if the file has changed
    file_hash = calculate_file_hash(uploaded_file)

    # Check if the uploaded file is different from the last file
    if file_hash != st.session_state["last_file_hash"]:
        # Save the uploaded file to a static location in `data`
        with open(SAVED_FILE_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load data using the saved file path
        loader = CSVLoader(file_path=SAVED_FILE_PATH, encoding="utf-8", csv_args={'delimiter': ','})
        data = loader.load()

        # Split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(data)

        # Initialize embeddings and FAISS database
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        docsearch = FAISS.from_documents(text_chunks, embeddings)
        docsearch.save_local(DB_FAISS_PATH)
        
        # Initialize the LLM
        llm = CTransformers(
            model="llama-2-7b-chat.ggmlv3.q8_0.bin",
            model_type="llama",
            config={'max_new_tokens': 1024, 'temperature': 0.1, 'context_length': 2048}
        )
        
        # Set up the Conversational Retrieval Chain
        st.session_state["qa"] = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

        # Update session state to mark database as initialized and store the new file hash
        st.session_state["db_initialized"] = True
        st.session_state["last_file_hash"] = file_hash
        st.success("Database initialized. Start chatting!")
    else:
        st.info("Same file detected, using existing database.")

# Display conversation history at the top
if st.session_state["db_initialized"]:
    # Display chat history messages
    for i, (query, response) in enumerate(st.session_state["chat_history"]):
        message(query, is_user=True, key=f"{i}_user")
        message(response, is_user=False, key=f"{i}_bot")

    # Chat input at the bottom
    user_input = st.text_input("Ask a question about your data:", "")

    # If user submits input, process it
    if user_input and not st.session_state["user_input_submitted"]:
        # Process the user query using the stored `qa` chain
        result = st.session_state["qa"]({"question": user_input, "chat_history": st.session_state["chat_history"]})
        
        # Append the user's query and the model's response to the chat history
        st.session_state["chat_history"].append((user_input, result["answer"]))
        
        # Indicate that user input was submitted to avoid re-processing
        st.session_state["user_input_submitted"] = True

        # Clear the input field by resetting `user_input_submitted` on rerun
        st.experimental_rerun()

    # Reset input submission state for next query
    st.session_state["user_input_submitted"] = False

else:
    st.info("Upload a CSV file to start chatting.")
