import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import tempfile
import time

# --- Load environment variables ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- Streamlit setup ---
st.set_page_config(page_title="💬 Persistent DocChat", layout="wide")
st.title("💬 Persistent Chat with Your Documents")

# --- Persistent Vector DB location ---
CHROMA_PATH = "chroma_db"

@st.cache_resource
def load_base_pipeline():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load or create Chroma DB
    if os.path.exists(CHROMA_PATH):
        st.info("✅ Loading existing Chroma DB...")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    else:
        st.warning("⚠️ No Chroma DB found yet. Upload files to create one.")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

    # Use memory that summarizes conversation context
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
        max_token_limit=1000,
    )

    prompt_template = """
    You are a helpful assistant. Use the context below to answer questions accurately.
    Context: {context}
    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    retriever = db.as_retriever(search_kwargs={"k": 3})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )

    return qa_chain, db, embeddings


qa_chain, db, embeddings = load_base_pipeline()

# --- File upload section ---
uploaded_files = st.file_uploader("📂 Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for uploaded_file in uploaded_files:
        file_ext = uploaded_file.name.split(".")[-1].lower()

        # Write safely to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp.flush()
            os.fsync(tmp.fileno())  # ensures file is fully written before reading
            tmp_path = tmp.name

        # Ensure file is not empty
        if os.path.getsize(tmp_path) == 0:
            st.error(f"🚫 The uploaded file `{uploaded_file.name}` is empty — skipping.")
            continue

        # Small wait to avoid Windows file locks
        time.sleep(0.2)

        # Load PDF or text safely
        try:
            if file_ext == "pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)

            new_docs = loader.load()
            new_chunks = splitter.split_documents(new_docs)
            docs.extend(new_chunks)

        except Exception as e:
            st.error(f"⚠️ Failed to process {uploaded_file.name}: {str(e)}")
            continue

    # Add to Chroma and persist
    if docs:
        db.add_documents(docs)
        db.persist()
        st.success(f"✅ Added and persisted {len(docs)} document chunks!")
    else:
        st.warning("No valid documents were processed.")

# --- Chat section ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about your uploaded or existing documents..."):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke({"question": question})
                answer = result["answer"]
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})