import streamlit as st
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from io import BytesIO
from huggingface_hub import login

# Use your Hugging Face token
login(token="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")

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(uploaded_file):
    with BytesIO(uploaded_file.read()) as f:
        with pdfplumber.open(f) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
    return text
    
# Function to create chunks of extracted text
def create_chunks(extracted_data):
    documents = [Document(page_content=extracted_data)]  # Wrap the extracted text in Document objects
    
    # Now, split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Function to load the embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to load the LLM (Language Model) from Hugging Face
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": "", "max_length": "512"}
    )
    return llm

# Define the custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Function to set up the retrieval-based QA chain
def setup_qa_chain(db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm("mistralai/Mistral-7B-Instruct-v0.3"),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    return qa_chain

# Streamlit UI
st.title("PDF Chatbot App")

# Session state initialization
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# File upload section
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    st.write("Processing your PDF...")
    
    # Extract text from uploaded PDF
    documents = extract_text_from_pdf(uploaded_file)
    st.write(f"Extracted text from your PDF.")
    
    # Create chunks from the documents
    text_chunks = create_chunks(documents)
    st.write(f"Created {len(text_chunks)} text chunks.")
    
    # Embedding and FAISS storage
    embedding_model = get_embedding_model()
    db = FAISS.from_documents(text_chunks, embedding_model)
    DB_FAISS_PATH = "/kaggle/working/vectorstore/db_faiss"
    db.save_local(DB_FAISS_PATH)
    
    # Load the QA chain
    qa_chain = setup_qa_chain(db)

    # Display previous conversation
    if st.session_state.conversation:
        for q, a in st.session_state.conversation:
            st.write(f"**Q:** {q}")
            st.write(f"**A:** {a}")
    
    # User input for question
    user_query = st.text_input("Ask a question about the PDF:")

    if user_query:
        # Get the response from the QA system
        response = qa_chain.invoke({'query': user_query})
        answer = response["result"]
        
        # Display the new answer
        st.write("### Answer:")
        st.write(answer)
        
        # Save the conversation
        st.session_state.conversation.append((user_query, answer))  # Store both question and answer

