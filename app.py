import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize embeddings (with timeout + retries)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,
    request_timeout=60,
    max_retries=5
)

# Initialize LLM
llm = OpenAI(
    temperature=0.9,
    max_tokens=500,
    openai_api_key=OPENAI_API_KEY
)

# Streamlit UI
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"  # file path for FAISS index

main_placeholder = st.empty()

# If user clicks "Process URLs"
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create FAISS vectorstore
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    pkl = vectorstore_openai.serialize_to_bytes()
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save FAISS index to pickle file
    with open(file_path, "wb") as f:
        pickle.dump(pkl, f)

# User Query input
query = main_placeholder.text_input("Question:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            pkl = pickle.load(f)
            # Load FAISS index
            vectorstore = FAISS.deserialize_from_bytes(
                embeddings=embeddings,
                serialized=pkl,
                allow_dangerous_deserialization=True
            )
            # Retrieval QA chain
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            result = chain({"question": query}, return_only_outputs=True)

            # Display Answer
            st.header("Answer")
            st.write(result["answer"])

            # Display Sources
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
