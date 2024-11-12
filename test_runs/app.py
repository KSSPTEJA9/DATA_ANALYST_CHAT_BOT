import streamlit as st 
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from sentence_transformers import SentenceTransformer

DB_FAISS_PATH = 'vectorstore/db_faiss'

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 1024,
        temperature = 0.5,
        context_length=2048
    )
    return llm

st.title("Chat with CSV using Llama2 🦙🦜")

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    # Use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    # Use HuggingFaceEmbeddings for embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

    # Create FAISS vector store
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)

    # Load LLM and create ConversationalRetrievalChain
    llm = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    # Initialize the tokenizer for chunking
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def chunk_input(input_text, max_chunk_size=512):
        """
        Chunk the input text into smaller pieces that fit within the token limit.
        """
        # Tokenize the input text
        tokens = tokenizer.encode(input_text)
        chunks = []
        
        # Split tokens into smaller chunks
        for i in range(0, len(tokens), max_chunk_size):
            chunk = tokens[i:i + max_chunk_size]
            # Decode chunk back to text
            chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
        
        return chunks

    def conversational_chat(query):
        # Handle chunking with consideration of chat history
        chat_history = st.session_state.get('history', [])
        chat_history_text = " ".join([f"User: {q[0]}\nAI: {q[1]}" for q in chat_history])

        # Prepare the full input including query and chat history
        full_input = f"{chat_history_text}\nUser: {query}"
        
        # Calculate the maximum number of tokens allowed for history
        remaining_tokens = 2048 - len(tokenizer.encode(query))  # Remaining tokens for history after the query
        chat_history_token_count = len(tokenizer.encode(chat_history_text))

        # Limit the history length to avoid exceeding context length
        if chat_history_token_count > remaining_tokens:
            # Limit chat history
            while chat_history_token_count > remaining_tokens:
                chat_history.pop(0)  # Remove the oldest interaction
                chat_history_token_count = len(tokenizer.encode(" ".join([f"User: {q[0]}\nAI: {q[1]}" for q in chat_history])))

        # Rebuild the history text
        chat_history_text = " ".join([f"User: {q[0]}\nAI: {q[1]}" for q in chat_history])
        full_input = f"{chat_history_text}\nUser: {query}"

        # Chunk the input
        chunks = chunk_input(full_input)

        results = []
        
        # Process each chunk and collect results
        for chunk in chunks:
            result = chain.invoke({"question": chunk, "chat_history": chat_history})
            results.append(result["answer"])
        
        # Combine results
        combined_result = " ".join(results)
        st.session_state['history'].append((query, combined_result))
        return combined_result

    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]
        
    # Container for the chat history
    response_container = st.container()
    # Container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")