import streamlit as st
from streamlit_chat import message
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
import os
import uuid

# Paths for the FAISS database and model
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_FOLDER_PATH = "data"

# Ensure the data folder exists
os.makedirs(DATA_FOLDER_PATH, exist_ok=True)

# Initialize session states for chat history and database
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "db_initialized" not in st.session_state:
    st.session_state["db_initialized"] = False

# Streamlit UI components
st.title("Chat with Your CSV Data 📊")

# File upload component
uploaded_file = st.file_uploader("Upload a CSV file to chat with", type="csv")

# Initialize FAISS and other components if a file is uploaded
if uploaded_file is not None:
    # Generate a unique filename using UUID
    unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
    temp_file_path = os.path.join(DATA_FOLDER_PATH, unique_filename)

    # Save the uploaded file to the data folder with a unique name
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load data using the saved file path
    loader = CSVLoader(file_path=temp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
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
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

    # Set database initialized to True
    st.session_state["db_initialized"] = True
    st.success("Database initialized. Start chatting!")

# Check if database is ready
if st.session_state["db_initialized"]:
    # Chat input
    user_input = st.text_input("Ask a question about your data:", "")
    
    if user_input:
        # Process the user query and add to chat history
        result = qa({"question": user_input, "chat_history": st.session_state["chat_history"]})
        
        # Append the user's query and the model's response to the chat history
        st.session_state["chat_history"].append((user_input, result["answer"]))
        
        # Display conversation history
        for i, (query, response) in enumerate(st.session_state["chat_history"]):
            message(query, is_user=True, key=f"{i}_user")
            message(response, is_user=False, key=f"{i}_bot")

else:
    st.info("Upload a CSV file to start chatting.")
