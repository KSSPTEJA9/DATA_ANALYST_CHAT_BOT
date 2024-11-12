from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import sys
import pandas as pd
from sentence_transformers import CrossEncoder


cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')

# Load and prepare document data
DB_FAISS_PATH = "vectorstore/db_faiss"
file_path="data/adae1.csv"
loader = CSVLoader(file_path="data/adae1.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
print("Data loaded:", data)

# Split the text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
print("Number of chunks:", len(text_chunks))

# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# Converting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(text_chunks, embeddings)
docsearch.save_local(DB_FAISS_PATH)

# Initialize LLM
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin",
                    model_type="llama",
                    config = {'max_new_tokens':1024,
                              'temperature':0.1,
                              'context_length':2048 })

# Initialize Memory with specific input key
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",  # Specify that only "question" should be saved to memory
    return_messages=True
)

# Define a Self-Aware Prompt Template
prompt_template = """
You are a document-specific assistant. Use only the provided document excerpts to answer questions.

Step 1: Understand the user's question by identifying key entities, metrics, countries, or years in their question.
Step 2: Search the document for relevant excerpts using these entities.
Step 3: If you find relevant information, answer in natural language based only on these excerpts.
If the question asks for a calculation (like sum, average, etc.), calculate the result based on the relevant data extracted from the documents.
If you cannot find an answer in the document, respond with "I don't know based on the document."


Question: {question}
Context:
{context}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=prompt_template
)

# Create Conversational Retrieval Chain with Custom Prompt and Memory
qa = ConversationalRetrievalChain.from_llm(
    llm, 
    retriever=docsearch.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}  # Pass custom prompt to limit scope
)

# def interpret_question(query):
#     # Basic question interpretation (can be replaced with advanced NLP processing)
#     keywords = [word.strip() for word in query.split() if len(word) > 3]
#     return " ".join(keywords)
# import spacy
# nlp = spacy.load("en_core_web_sm")

# # Load the CSV and get column names
# def get_column_names_from_csv(file_path):
#     df = pd.read_csv(file_path)
#     return set(df.columns.str.lower())  # Using lowercase to standardize matching

# # Interpret the question with dynamic column name checks
# def interpret_question(query, file_path):
#     # Get the column names
#     column_names = get_column_names_from_csv(file_path)
    
#     # Parse the question with spaCy
#     doc = nlp(query)
    
#     # Collect relevant keywords
#     keywords = []

#     # Extract named entities if they match any column name
#     for ent in doc.ents:
#         if ent.text.lower() in column_names:
#             keywords.append(ent.text)
    
#     # If no entities are found, fall back to nouns and key terms
#     if not keywords:
#         for token in doc:
#             if token.pos_ in {"NOUN", "PROPN", "NUM"} and not token.is_stop:
#                 if token.text.lower() in column_names:
#                     keywords.append(token.text)
    
#     # Join keywords into a refined query
#     refined_query = " ".join(keywords)
#     return refined_query

refine_prompt = """
You are a helpful assistant. Based on the user's question, extract the most important terms.
Identify the country, year, metric (e.g., GDP, population), and any comparison terms (e.g., highest, lowest, max, min).
The question might be asking for information such as the highest or lowest value of a metric in a specific year.

Question: "{question}"
Output: A refined query with extracted terms: country, year, metric, and comparison terms.
"""

import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load spaCy model for NLP processing
nlp = spacy.load("en_core_web_sm")

# Load CSV and get column names
def get_column_names_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df.columns.str.lower().tolist()  # Standardize to lowercase for consistent matching

# Function to extract relevant terms dynamically from the query
def extract_relevant_terms(query):
    doc = nlp(query.lower())  # Process the query text with spaCy
    terms = set()

    # Extract entities and key tokens
    for ent in doc.ents:
        terms.add(ent.text)
    
    # Also, extract important nouns or keywords that are not entities
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "NUM"} and not token.is_stop:
            terms.add(token.text)

    return terms

# Function to find the best matching column using semantic similarity
def find_best_column_match(query, column_names):
    # Encode the query and column names to embeddings
    query_embedding = model.encode(query, convert_to_tensor=True)
    column_embeddings = model.encode(column_names, convert_to_tensor=True)

    # Compute cosine similarities between the query and column names
    cosine_scores = util.pytorch_cos_sim(query_embedding, column_embeddings)

    # Find best match for the query
    best_match_idx = cosine_scores.argmax()
    return column_names[best_match_idx]

# Interpret the question with dynamic column name checks
def interpret_question(query, file_path):
    # Get the column names from the CSV
    column_names = get_column_names_from_csv(file_path)
    
    # Extract relevant terms from the query
    relevant_terms = extract_relevant_terms(query)

    # Find best matches for each extracted term
    matches = {}
    for term in relevant_terms:
        best_match = find_best_column_match(term, column_names)
        matches[term] = best_match

    # Return the matched columns as the refined query
    refined_query = " ".join(matches.values())
    return refined_query


def get_best_matching_document(query, docs):
    # Prepare the input for the Cross-Encoder
    inputs = [[query, doc] for doc in docs]
    scores = cross_encoder.predict(inputs)
    best_doc_idx = scores.argmax()
    return docs[best_doc_idx]


# Enhanced Chat loop with question understanding and document-specific responses
while True:
    query = input("Input Prompt: ")
    if query.lower() == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    
    # Interpret the question to refine the search
    refined_query = interpret_question(query,file_path=file_path)
    print(refined_query)
    # Perform document-specific retrieval based on interpreted question
    docs = docsearch.similarity_search(query, k=5)
    documents = [doc.page_content for doc in docs]

    #re ranking

    #best_document = get_best_matching_document(query, documents)
    

    

    context = "\n".join([doc.page_content for doc in docs])  # Combine retrieved docs for context
    #print(context)
    print(context)
    # Generate response using LLM with retrieved context
    result = qa.invoke({"question": query, "context": context})  # Use invoke method as suggested
    print("Response:", result['answer'])
