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

txt_file = '../data/dynamic_analysis_output.txt'

with open(txt_file, 'r') as f:
    content = f.read()

#llm = Ollama(model="deepseek-r1", temperature=0.2)

llm = ChatBedrock(
    model_id='us.anthropic.claude-3-5-haiku-20241022-v1:0',
    model_kwargs={"temperature": 0.2},
)

embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0')

system_prompt_template = """
You are a highly analytical agent specializing in both security and functional review. 
Your task is to prioritize previous analysis of HTTP requests based on their potential security risks.

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
Prioritize the following analysis of HTTP Requests based on their potential security risks, output all endpoints that are potentially vulnerable to injection attacks

<content>
{content}
</content>

ONLY respond with the following information:
- URL: (str) The full URL of the request in the format: http://example.com/path
- Potential Severity: (str) The severity of the potential vulnerability (e.g., High, Medium, Low)
- HTTP Method: (str) The HTTP Method of the request
- Parameters: (str) The parameters of the request
- Possible Injection: (str) Yes or No
- Justification: (str) A brief justification ONLY if injection exploit may be possible
- Test Instructions: (str) Instructions on how to test the endpoint for vulnerabilities

DO NOT PROVIDE ADDITIONAL INFORMATION.
"""

chain = (
    { "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in chain.stream({"question": question, "content": content}):
    print(chunk, end="", flush=True)

print("=" * 50)
