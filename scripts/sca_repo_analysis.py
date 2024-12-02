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



faiss_db_path = "../vector_databases/repo_scan_results_faiss"
db = FAISS.load_local(
    faiss_db_path, 
    BedrockEmbeddings(model_id='amazon.titan-embed-text-v1'),
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_type="mmr", # Also test "similarity"
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
                ("human", """<question>{question}</question>""")
            ]
)

llm = ChatBedrock(
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
    model_kwargs={"temperature": 0.6},
)

chain = (
     {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

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
"""

for chunk in chain.stream(user_question):
                print(chunk, end="", flush=True)