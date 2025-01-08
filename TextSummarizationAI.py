import streamlit as st
from langchain.chains import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter  # Import splitter
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import torch


# Initialize the summarization model (T5 or DistilBART)
@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1  # Dynamically select GPU or CPU
    device = -1  # Force CPU
    summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    return HuggingFacePipeline(pipeline=summarizer_pipeline)


# Load LangChain with summarization model
llm = load_model()

# Streamlit App
st.title("Summarize and Extract Answers")

uploaded_file = st.file_uploader("Upload your text file", type=["txt", "md"])

if uploaded_file is not None:
    # Read the uploaded file
    text_data = uploaded_file.read().decode("utf-8")
    st.subheader("Uploaded Text")
    st.text_area("File Content", text_data, height=200)

    # Split the text into manageable chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text_data)
    documents = [Document(page_content=chunk) for chunk in chunks]

    st.write(f"Document split into {len(documents)} chunks.")

    # Summarize the data
    if st.button("Summarize"):
        st.info("Summarizing... Please wait.")
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.invoke({"input_documents": documents})
        st.subheader("Summary")
        st.write(summary["output_text"])  # Extract summary text

        # Ask a question about the text
    user_question = st.text_input("Ask a question about the text")
    if user_question:
        st.info("Searching for an answer...")
        relevant_chunks = []

        # Filter chunks for relevance to the question
        for chunk in chunks:
            if user_question.lower() in chunk.lower():
                relevant_chunks.append(chunk)

        # Use only relevant chunks or truncate to fit token limits
        context = " ".join(relevant_chunks[:2]) if relevant_chunks else chunks[0]

        # Create a question-answering prompt
        qa_prompt = PromptTemplate(
            input_variables=["text", "question"],
            template="Based on the following text: {text}\nAnswer the question: {question}"
        )
        formatted_prompt = qa_prompt.format(text=context, question=user_question)

        # Pass the prompt to the model
        try:
            answer = llm(formatted_prompt)
            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error while processing your question: {e}")
