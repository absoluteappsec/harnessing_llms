import os
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from langchain_ollama import OllamaLLM as Ollama

import base64
import xml.etree.ElementTree as ET

# Load Env Variables
from dotenv import load_dotenv
load_dotenv()

#llm = Ollama(model="deepseek-r1", temperature=0.2)

llm = ChatBedrock(
    model_id='us.anthropic.claude-3-5-haiku-20241022-v1:0',
    model_kwargs={"temperature": 0.2},
)

embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0')

faiss_db_path = "../vector_databases/vtm_session.faiss"
db = FAISS.load_local(
    faiss_db_path, 
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_type="mmr", # Also test "similarity"
    search_kwargs={"k": 8},
)

system_prompt_template = """
You are a highly analytical agent specializing in both security and functional review. 
Your task is to analyze an HTTP Request for user-controllable parameters that could be used for injection exploits.

Context for analysis:
<context>
{context}
</context>

Remember to:
- Identify areas where more investigation might be needed
- Only output the requested information, do not provide any additional details.

"""


prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_template),
                ("human", """<question>{question}</question>""")
            ]
)

question = """"
Please analyze the full HTTP Session for possibility user-controlled parameters that could be used for injection exploits such as SQL Injection, Command Injection, or other types of injection attacks:

ONLY respond with the following information:
- URL: (str) The full URL of the request in the format: http://example.com/path
- HTTP Method: (str) The HTTP Method of the request
- Parameters: (str) The parameters of the request
- Possible Injection: (str) Yes or No
- Justification: (str) A brief justification ONLY if injection exploit may be possible

DO NOT PROVIDE ADDITIONAL INFORMATION.

Analyze each request in the session until all requests have been analyzed.
"""

chain = (
     {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in chain.stream(question):
                print(chunk, end="", flush=True)
