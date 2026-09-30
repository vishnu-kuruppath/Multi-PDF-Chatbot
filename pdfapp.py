import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_classic.chains import RetrievalQA

import tempfile

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Multi PDF Chatbot (RAG + Ollama)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)


if uploaded_files and st.button("Process Documents"):
    with st.spinner("Processing PDFs..."):

        all_docs = []

        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            all_docs.extend(docs)

        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        llm = OllamaLLM(model="gemma3:1b")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_db.as_retriever(),
            chain_type="stuff"
        )

        st.session_state.qa_chain = qa_chain
        st.success("✅ Documents processed successfully!")


if st.session_state.qa_chain:
    question = st.text_input("Ask a question:")

    if question:
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke(question)

            answer_text = response.get("result", "")

            if answer_text.strip() == "" or "i don't know" in answer_text.lower():
                answer_text = "Answer not found in the uploaded documents."

            st.session_state.chat_history.append(("You", question))
            st.session_state.chat_history.append(("Bot", answer_text))

st.subheader("Chat History")
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.write(f"🧑 **You:** {msg}")
    else:
        st.write(f"🤖 **Bot:** {msg}")