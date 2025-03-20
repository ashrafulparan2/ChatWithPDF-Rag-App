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
import streamlit.components.v1 as components

# Log in to Hugging Face
login(token="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")

# Function definitions (unchanged)
def extract_text_from_pdf(uploaded_file):
    with BytesIO(uploaded_file.read()) as f:
        with pdfplumber.open(f) as pdf:
            text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
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

# Particles.js animation (from Exifa.net)
particles_js = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Particles.js</title>
  <style>
  #particles-js {
    position: fixed;
    width: 100vw;
    height: 100vh;
    top: 0;
    left: 0;
    z-index: -1;
  }
  .content {
    position: relative;
    z-index: 1;
    color: white;
  }
  </style>
</head>
<body>
  <div id="particles-js"></div>
  <div class="content"></div>
  <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
  <script>
    particlesJS("particles-js", {
      "particles": {
        "number": {"value": 300, "density": {"enable": true, "value_area": 800}},
        "color": {"value": "#ffffff"},
        "shape": {"type": "circle"},
        "opacity": {"value": 0.5, "random": false},
        "size": {"value": 2, "random": true},
        "line_linked": {"enable": true, "distance": 100, "color": "#ffffff", "opacity": 0.22, "width": 1},
        "move": {"enable": true, "speed": 0.2, "direction": "none", "random": false, "straight": false, "out_mode": "out", "bounce": true}
      },
      "interactivity": {
        "detect_on": "canvas",
        "events": {"onhover": {"enable": true, "mode": "grab"}, "onclick": {"enable": true, "mode": "repulse"}, "resize": true},
        "modes": {"grab": {"distance": 100, "line_linked": {"opacity": 1}}, "repulse": {"distance": 200, "duration": 0.4}}
      },
      "retina_detect": true
    });
  </script>
</body>
</html>
"""

# Streamlit Configuration
st.set_page_config(page_title="PDF Chatbot", page_icon="📜", layout="wide")

# Icons for chat
icons = {
    "assistant": "https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/assistant.gif",
    "user": "https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/user.gif",
}

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a PDF to start chatting."}]
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "show_animation" not in st.session_state:
    st.session_state.show_animation = True

# Sidebar
with st.sidebar:
    st.markdown(
        f"""
        <div style='display: flex; align-items: center;'>
            <img src='https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/Exifa.gif' style='width: 50px; height: 50px; margin-right: 30px;'>
            <h1 style='margin: 0;'>PDF Chatbot</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🗀 File Input"):
        uploaded_file = st.file_uploader("Upload a PDF to chat with", type=["pdf"], key="pdf_uploader")
    
    with st.expander("⚒ Model Configuration"):
        st.subheader("Adjust model parameters")
        temperature = st.slider("Temperature", min_value=0.01, max_value=5.0, value=0.5, step=0.01)
        max_length = st.number_input("Max Length", min_value=128, max_value=1024, value=512)

    st.caption(
        "Built by [Ashraful Islam Paran](https://www.linkedin.com/in/ashrafulparan/)."
    )
    st.caption(
        f"""
        <div style='display: flex; align-items: center;'>
            <a href='https://www.linkedin.com/in/ashrafulparan/'><img src='https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/linkedin.gif' style='width: 35px; height: 35px; margin-right: 25px;'></a>
            <a href='mailto:ashrafulparan@gmail.com'><img src='https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/email.gif' style='width: 28px; height: 28px; margin-right: 25px;'></a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main content
if uploaded_file is not None and st.session_state.qa_chain is None:
    with st.spinner("Processing your PDF..."):
        documents = extract_text_from_pdf(uploaded_file)
        text_chunks = create_chunks(documents)
        embedding_model = get_embedding_model()
        db = FAISS.from_documents(text_chunks, embedding_model)
        DB_FAISS_PATH = "db_faiss"
        db.save_local(DB_FAISS_PATH)
        st.session_state.qa_chain = setup_qa_chain(db)
    st.success("PDF processed! You can now chat with it.")
    st.session_state.messages = [{"role": "assistant", "content": "PDF processed! Ask me anything about it."}]

# Chat display
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=icons[message["role"]]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the PDF:"):
    st.session_state.show_animation = False
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=icons["user"]):
        st.write(prompt)

    if st.session_state.qa_chain:
        with st.chat_message("assistant", avatar=icons["assistant"]):
            response = st.session_state.qa_chain.invoke({'query': prompt})
            answer = response["result"]
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Animation
if st.session_state.show_animation:
    components.html(particles_js, height=370, scrolling=False)

# Clear chat history
if st.button("🗑 Clear Chat History"):
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a PDF to start chatting."}]
    st.session_state.qa_chain = None
    st.session_state.show_animation = True
    st.success("Chat history cleared!")