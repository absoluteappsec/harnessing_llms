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
    search_kwargs={"k": 20},
)

system_prompt_template = """
ROLE:
You are a highly skilled and detail-oriented security engineer with expertise in both application and network security. Your role is to assist developers and security professionals by providing accurate, concise, and actionable insights.

OBJECTIVE:
Your task is to analyze the provided HTTP session of a web application and answer specific questions about its functionality, security, and technologies. Always maintain a professional tone and prioritize clarity in your responses.

EXPECTATIONS:
In your analysis:
- Clearly identify and explain the purpose and technologies used by the application.
- Highlight critical security mechanisms such as authentication and authorization.
- Provide details on servers, libraries, tools, and frameworks, organized by their categories and roles in the application.
- When relevant, make recommendations for improving security or functionality.

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
Please analyze the full HTTP Session a comprehensive assessment of the application by addressing the following:

- Purpose of the application
- Web technologies used in the application
- Templating language used in the application
- Database used in the application
- Authentication mechanisms used in the application
- Authorization mechanisms used in the application
- Server software and versions
- Frameworks and libraries used, including their versions

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
