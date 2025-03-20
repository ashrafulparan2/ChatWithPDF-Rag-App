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

# Function definitions remain the same
def extract_text_from_pdf(uploaded_file):
    with BytesIO(uploaded_file.read()) as f:
        with pdfplumber.open(f) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
    return text

def create_chunks(extracted_data):
    documents = [Document(page_content=extracted_data)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": "", "max_length": "512"}
    )
    return llm

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't provide anything out of the given context

Context: {context}
Question: {question}
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def setup_qa_chain(db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm("mistralai/Mistral-7B-Instruct-v0.3"),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    return qa_chain

# Streamlit UI with Conversational Style
st.title("PDF Chatbot")

# Custom CSS for chat-like appearance
st.markdown("""
    <style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #ccc;
        padding: 10px;
        border-radius: 5px;
        background-color: #f9f9f9;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 70%;
        align-self: flex-end;
    }
    .bot-message {
        background-color: #e9ecef;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 70%;
    }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# File upload section
uploaded_file = st.file_uploader("Upload a PDF to chat with", type=["pdf"])

if uploaded_file is not None:
    if st.session_state.qa_chain is None:
        with st.spinner("Processing your PDF..."):
            # Extract text from uploaded PDF
            documents = extract_text_from_pdf(uploaded_file)
            
            # Create chunks from the documents
            text_chunks = create_chunks(documents)
            
            # Embedding and FAISS storage
            embedding_model = get_embedding_model()
            db = FAISS.from_documents(text_chunks, embedding_model)
            DB_FAISS_PATH = "db_faiss"
            db.save_local(DB_FAISS_PATH)
            
            # Load the QA chain
            st.session_state.qa_chain = setup_qa_chain(db)
        st.success("PDF processed! You can now chat with it.")

    # Chat container
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for q, a in st.session_state.conversation:
            st.markdown(f'<div class="user-message">{q}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-message">{a}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # User input for question
    with st.form(key='chat_form', clear_on_submit=True):
        user_query = st.text_input("Ask a question about the PDF:", key="user_input")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_query:
        # Get the response from the QA system
        response = st.session_state.qa_chain.invoke({'query': user_query})
        answer = response["result"]
        
        # Save the conversation
        st.session_state.conversation.append((user_query, answer))
        
        # Rerun to update the chat display
        st.rerun()

else:
    st.info("Please upload a PDF to start chatting.")