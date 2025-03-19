import os
import pinecone
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Pinecone
from langchain_cohere.llms import Cohere
from langchain_cohere import CohereEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from pinecone import ServerlessSpec  # ✅ Import required for Pinecone spec

# ✅ Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp-starter")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise ValueError(" ERROR: COHERE_API_KEY is missing. Set it in .env or environment variables.")

if not PINECONE_API_KEY:
    raise ValueError(" ERROR: PINECONE_API_KEY is missing. Set it in .env or environment variables.")

# ✅ Initialize FastAPI App
app = FastAPI()

# ✅ Initialize Pinecone Client
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# ✅ Define Pinecone Index Name
index_name = "medical-chatbot"

# ✅ Check if Index Exists, If Not, Create One
if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=index_name,
        dimension=1024,  # ✅ Adjust based on Cohere embeddings
        metric="cosine",
        spec=ServerlessSpec(  # ✅ Corrected Spec Argument
            cloud="gcp",  # or "gcp"
            region="starter"
        )
    )

# ✅ Connect to Pinecone Index
index = pinecone_client.Index(index_name)

# ✅ Load PDF Documents
pdf_loader = PyPDFLoader("medical_book.pdf")
docs = pdf_loader.load()

# ✅ Split Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# ✅ Use Cohere Embeddings
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

# ✅ Store Embeddings in Pinecone
vectorstore = Pinecone.from_documents(chunks, embeddings, index_name=index_name)

retriever = vectorstore.as_retriever()

# ✅ Use Cohere as LLM
llm = Cohere(model="command", cohere_api_key=COHERE_API_KEY)

# ✅ Set Up Conversational Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# ✅ Define API Request Model
class UserQuery(BaseModel):
    query: str

# ✅ API Endpoint
@app.post("/chat")
async def chat(user_query: UserQuery):
    try:
        response = qa_chain.run(user_query.query)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

# ✅ Run using: `uvicorn main:app --reload`
