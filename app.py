import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Function to extract text from .txt files
def get_txt_text(txt_files):
    text = ""
    for txt in txt_files:
        text += txt.read().decode("utf-8")  # Reading and decoding text files
    return text

# Function to split text into chunks
def get_text_chunks(text):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question only in one or two  word as possible from the provided context. If the answer is not in
    the provided context, just say "answer is not available in the context". Do not provide an incorrect answer.
    
    Context:\n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user question input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("Reply: ", response["output_text"])

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with TXT Files", layout="wide")
    st.header("Chat with Plain Text Files using Gemini 💁")

    user_question = st.text_input("Ask a question based on the uploaded text files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        txt_files = st.file_uploader("Upload your .txt files", accept_multiple_files=True, type=["txt"])

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if txt_files:
                    raw_text = get_txt_text(txt_files)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done processing the text files!")
                else:
                    st.warning("Please upload valid text files!")

if __name__ == "__main__":
    main()
