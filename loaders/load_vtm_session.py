import os
import git


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document

# Load Env Variables
from dotenv import load_dotenv
load_dotenv()

# For BedRock
from langchain_aws import BedrockEmbeddings

import base64
import xml.etree.ElementTree as ET

xml_file = 'data/vtm-session.xml'

xml = ET.parse(xml_file)
root = xml.getroot()

print(f"Parsing {len(root)} requests")

embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0')

documents = []
count = 1
for item in root:
    print(f"=> {count}/{len(root)}: {item.find('url').text}")
    request = item.find("request").text
    if item.find("request").attrib['base64'] == 'true':
        request = base64.b64decode(request).decode('utf-8')
    response = item.find("response").text
    if item.find("response").attrib['base64'] == 'true':
        response = base64.b64decode(response).decode('utf-8')
    content = f"{request}\n\n{response}"
    documents.append(
        Document(
            page_content=content,
            metadata={
                "id": count,
                "method": item.find("method").text,
                "url": item.find("url").text,
            }
        )
    )
    count += 1
        

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8000, chunk_overlap=100
)
texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} chunks")
# Create FAISS vector store from the documents
db = FAISS.from_documents(texts, embeddings)
db.save_local("vector_databases/vtm_session.faiss")