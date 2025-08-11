import os
import git
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load Env Variables
from dotenv import load_dotenv

load_dotenv()

repo_url = "https://github.com/redpointsec/vtm.git"
local_path = "./repo"

if os.path.isdir(local_path) and os.path.isdir(os.path.join(local_path, ".git")):
    print("Directory already contains a git repository.")
else:
    try:
        repo = git.Repo.clone_from(repo_url, local_path)
        print(f"Repository cloned into: {local_path}")
    except Exception as e:
        print(f"An error occurred while cloning the repository: {e}")


# THIS LOADS ONLY PYTHON EXTENSION FILES, CHANGE AS NEEDED
python_files = {}
# Traverse the directory recursively
for root, _, files in os.walk(local_path):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            try:
                # Read the contents of the Python file
                with open(file_path, "r", encoding="utf-8") as f:
                    python_files[file_path] = f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")


llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    model_kwargs={"temperature": 0.2},
)

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

system_prompt_template = """
You are a helpful code review assistant who is 
proficient in both security as well as functional review. 
You will be provided source code of a web application and 
tasked with answering questions about it.

<context>
{context}
</context>
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("human", """<question>{question}</question>"""),
    ]
)

question = """"
Analyze the provided code for any security 
flaws you find in it and produce a summary of that analysis.
"""

# Now we will iterate over the Python files and analyze them
# and store the results for later processing
docs = []
for file_path, content in python_files.items():
    response_array = []
    code = content
    # Retrieve filename for reference
    filename = file_path
    # Create a chain of operations to run the code through
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # This is an optional addition to stream the output in chunks
    # for a chat-like experience
    title = f"\n\nAnalyzing code from {filename}"
    print(title)
    print("=" * len(title))
    for chunk in chain.stream({"question": question, "context": code}):
        print(chunk, end="", flush=True)
        response_array.append(chunk)

    flattened_response = "".join(response_array)
    document = Document(
        page_content=flattened_response, metadata={"filename": filename}
    )

    docs.append(document)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=100)

# CHANGE AS DESIRED
name_of_scan_results_db = "repo_scan_results_faiss"

texts = text_splitter.split_documents(docs)
db = FAISS.from_documents(texts, embeddings)
db.save_local(f"../vector_databases/{name_of_scan_results_db}")
