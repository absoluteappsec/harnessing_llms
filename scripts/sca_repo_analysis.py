from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load Env Variables
from dotenv import load_dotenv

load_dotenv()

# For BedRock
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings

# CHANGE AS NEEDED
name_of_faiss_db = "repo_scan_results_faiss"

faiss_db_path = f"../vector_databases/{name_of_faiss_db}"
db = FAISS.load_local(
    faiss_db_path,
    BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0"),
    allow_dangerous_deserialization=True,
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 100},
)

system_prompt_template = """
You are a highly skilled pplication security expert. 
Your role is to review the results of scan tools and highlight
the most critical  developers and security professionals 
by providing accurate, concise, and actionable insights.

Your task is to analyze the results of a security source code
analysis tool and provide detailed explanations of the findings.
provided source code of a web application

Use the following context to help answer questions:
<context>
{context}
</context>

Respond in the following format:
- Vulnerability Type: (str) The type of vulnerability found.
- Vulnerability Description: (str) A brief description of the vulnerability.
- File Path: (str) The path to the file containing the vulnerability.
"""

# CORRECT/FORMAL WAY TO PERFORM PROMPTING
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("human", """<question>{question}</question>"""),
    ]
)

llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    model_kwargs={"temperature": 0.6},
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# CHANGE AS DESIRED
user_question = """
Tell me about only the highest criticality security 
vulnerabilities found in the codebase. The categories should
loosely pertain to:

- Injection
- Broken Authentication
- Sensitive Data Exposure
- Unsafe deserialization
- Remote Code Execution
- Hardcoded Secrets
- CSRF
"""

for chunk in chain.stream(user_question):
    print(chunk, end="", flush=True)
