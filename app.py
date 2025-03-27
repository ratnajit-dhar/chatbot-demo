import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

# Define the path to your saved vector store
DB_FAISS_PATH = 'vectorstore.faiss'

# Load the vector store
@st.cache_resource
def load_vectorstore():
    embedding = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-miniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding, allow_dangerous_deserialization=True)
    return db

# Load the Hugging Face Endpoint for Mistral 7B
@st.cache_resource
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": "", "max_length": "512"}
    )
    return llm

# Initialize Streamlit app
st.title("🤖 Green Lead Chatbot")
# st.write("Ask me anything about your knowledge base!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load the QA chain
@st.cache_resource
def load_qa_chain():
    db = load_vectorstore()
    retriever = db.as_retriever()
    llm = load_llm("mistralai/Mistral-7B-Instruct-v0.3")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Load the QA chain
qa_chain = load_qa_chain()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            response = qa_chain.run(prompt)
            message_placeholder.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# Add a clear button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Add some styling
st.markdown("""
    <style>
    .stChat {
        padding: 20px;
    }
    .stTextInput {
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)