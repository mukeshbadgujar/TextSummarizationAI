from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.schema import Document
from transformers import pipeline
import streamlit as st
import torch


@st.cache_resource
def load_qa_model():
    device = 0 if torch.cuda.is_available() else -1
    device = -1  # Force CPU
    return pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(text_chunks):
    """Build a vector store from the document chunks."""
    embeddings = load_embedding_model()
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store


# Load models
qa_model = load_qa_model()

# Streamlit App
st.title("Improved QA System")

uploaded_file = st.file_uploader("Upload your text file", type=["txt", "md"])

if uploaded_file is not None:
    text_data = uploaded_file.read().decode("utf-8")
    st.subheader("Uploaded Text")
    st.text_area("File Content", text_data, height=200)

    # Split text into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text_data)
    vector_store = build_vector_store(chunks)

    st.write(f"Document split into {len(chunks)} chunks.")

    # Ask a question about the text
    user_question = st.text_input("Ask a question about the text")
    if user_question:
        if not chunks:
            st.error("The text file is too small to extract answers.")
        else:
            st.info("Searching for an answer...")

            # Retrieve the most relevant chunks
            relevant_chunks = vector_store.similarity_search(user_question, k=5)
            context = " ".join([chunk.page_content for chunk in relevant_chunks])

            try:
                # Pass question and relevant context to the QA model
                result = qa_model({"question": user_question, "context": context})

                # Display the answer
                st.subheader("Answer")
                st.write(result["answer"])
                st.write(f"Confidence: {result['score']:.2f}")
            except Exception as e:
                st.error(f"Error during question answering: {e}")
