from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Load Env Variables
from dotenv import load_dotenv
load_dotenv()

# For BedRock
from langchain_aws import BedrockEmbeddings

embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1')

loader = PyPDFLoader(
    "../data/Acme_Co_Security_Guide.pdf",
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8000, chunk_overlap=100
)

texts = text_splitter.split_documents(documents)
db = FAISS.from_documents(texts, embeddings)
db.save_local("../vector_databases/acmeco_sec_guide_faiss")