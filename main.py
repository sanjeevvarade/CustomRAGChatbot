from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.chains import ConversationalRetrievalChain #added memory
#from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

# ✅ Set API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
print("GOOGLE_API_KEY loaded:", os.getenv("GOOGLE_API_KEY") is not None)

# Load pdf document
loader = PyPDFLoader("data/data.pdf")
pdfdoc = loader.load()

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# Load text document
textdoc = TextLoader('data/data.txt').load()

txtchunks = splitter.split_documents(textdoc)
pdfchunks = splitter.split_documents(pdfdoc)
chunks = txtchunks + pdfchunks

# Create embeddings and vector store
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#db = Chroma.from_documents(chunks, embeddings)

for doc in chunks:
    doc.metadata["source"] = getattr(doc, "source", "unknown")
#store meta data when creating metadata
db = Chroma.from_documents(chunks, embeddings)

# Create retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
response = llm.invoke("Say hello from Gemini!")
print(response)

# Create prompt template
prompt_template = """
You are a helpful assistant. Use the following context to answer the question accurately.
Context: {context}
Question: {question}
Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#added chat history/memory
#memory = ConversationBufferMemory(
#    memory_key="chat_history",  # key in chain input/output
#    return_messages=True       # stores full messages, not just text
#)

# to keeps a running summary of the conversation instead of full history
memory = ConversationSummaryMemory(
    llm=llm,                  # your Gemini LLM
    memory_key="chat_history", # key in chain input/output
    return_messages=True,      # stores full messages optionally
    input_key="question",      # incoming question key
    output_key="answer",       # chain output key
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT}
)

# Combine retrieval and QA
#qa_chain = RetrievalQA.from_chain_type(
#    llm=llm,
#    retriever=retriever,
#    chain_type_kwargs={"prompt": PROMPT}
#)

while True:
    question = input("\nAsk a question about your document (or 'exit'): ")
    if question.lower() == "exit":
        break
    result = qa_chain.invoke({"question": question})  #added memory
    print(f"\nAnswer: {result['answer']}")
    #result = qa_chain.invoke({"query": question})
    #print(f"\nAnswer: {result['result']}")
