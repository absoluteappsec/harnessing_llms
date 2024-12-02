import os
import git
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
# Load Env Variables
from dotenv import load_dotenv
load_dotenv()

# CHANGE THE REPO URL TO THE RELEVANT REPO URL
repo_url = 'https://github.com/railsbridge/bridge_troll.git'
local_path = './repo'

if os.path.isdir(local_path) and os.path.isdir(os.path.join(local_path, '.git')):
    print("Directory already contains a git repository.")
else:
    try:
        repo = git.Repo.clone_from(repo_url, local_path)
        print(f"Repository cloned into: {local_path}")
    except Exception as e:
        print(f"An error occurred while cloning the repository: {e}")

loader = GenericLoader.from_filesystem(
    local_path,
    glob="**/*",
    # CHANGE .rb TO THE RELEVANT FILE EXTENSION
    suffixes=[".rb"],
    # CHANGE Language.RUBY TO THE RELEVANT LANGUAGE
    parser=LanguageParser(language=Language.RUBY),
    show_progress=True
)   

documents = loader.load()

embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1')
# CHANGE Language.RUBY TO THE RELEVANT LANGUAGE
splitter = RecursiveCharacterTextSplitter.from_language(language=Language.RUBY, 
    chunk_size=8000, 
    chunk_overlap=100
)
texts = splitter.split_documents(documents)

# CHANGE THE DB NAME TO THE RELEVANT DB NAME
db_name = 'bridge_troll'

db = FAISS.from_documents(texts, embeddings)
db.save_local(f'../vector_databases/{db_name}_faiss')