import os
import git


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader

# Load Env Variables
from dotenv import load_dotenv

load_dotenv()

# For BedRock
from langchain_aws import BedrockEmbeddings


repo_url = "https://github.com/redpointsec/vtm.git"
repo_path = "./repo"

if os.path.isdir(repo_path) and os.path.isdir(os.path.join(repo_path, ".git")):
    print("Directory already contains a git repository.")
else:
    try:
        repo = git.Repo.clone_from(repo_url, repo_path)
        print(f"Repository cloned into: {repo_path}")
    except Exception as e:
        print(f"An error occurred while cloning the repository: {e}")


embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")


loader = DirectoryLoader(repo_path, glob="**/*.*", silent_errors=True)  # Load all files
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
db = FAISS.from_documents(texts, embeddings)
db.save_local("../vector_databases/vtm_faiss")
