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

xml_file = '../data/vtm-session.xml'

xml = ET.parse(xml_file)
root = xml.getroot()
print(f"Parsing {len(root)} requests")

#llm = Ollama(model="deepseek-r1", temperature=0.2)

llm = ChatBedrock(
    model_id='us.anthropic.claude-3-5-haiku-20241022-v1:0',
    model_kwargs={"temperature": 0.2},
)

embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0')

system_prompt_template = """
You are a highly analytical agent specializing in both security and functional review. 
Your task is to analyze an HTTP Request for user-controllable parameters that could be used for injection exploits.

Context for analysis:
{context}

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
Please analyze the following HTTP Request for possibility user-controlled parameters that could be used for injection exploits such as SQL Injection, Command Injection, or other types of injection attacks:

<content>
{content}
</content>

ONLY respond with the following information:
- URL: (str) The full URL of the request in the format: http://example.com/path
- HTTP Method: (str) The HTTP Method of the request
- Parameters: (str) The parameters of the request
- Possible Injection: (str) Yes or No
- Justification: (str) A brief justification ONLY if injection exploit may be possible

DO NOT PROVIDE ADDITIONAL INFORMATION.
"""

chain = (
    { "context": RunnablePassthrough() , "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

count = 1
urls = []
output = ""
for item in root:
    print(f"=> {count}/{len(root)}: {item.find('url').text}")
    output += f"Request {item.find('url').text}:\n"
    # Skip duplicate URLs, if needed
    #url = item.find('url').text
    #if url in urls:
    #    print("=> Duplicate URL, skipping")
    #    continue
    #urls.append(item.find('url').text)
    request = item.find("request").text
    if item.find("request").attrib['base64'] == 'true':
        request = base64.b64decode(request).decode('utf-8')
    count = count+1

    try: 
        answer = ""
        for chunk in chain.stream({"question": question, "content": request}):
            print(chunk, end="", flush=True)
            #response_array.append(chunk)
  
            answer += chunk
        output += answer + "\n\n"

        print("\n=> Complete\n")
    except Exception as e:
        print(f"=> Error: {e}")

# Save the output to a file
output_file = 'data/dynamic_analysis_output.txt'
with open(output_file, 'w') as f:
    f.write(output)
    f.close()
print(f"Output saved to {output_file}")
print("=" * 50)
