from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain.globals import set_debug

set_debug(True)

# Load Env Variables
from dotenv import load_dotenv
load_dotenv()

# For BedRock
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings



faiss_db_path = "../vector_databases/juice_shop.faiss"
db = FAISS.load_local(
    faiss_db_path, 
    BedrockEmbeddings(model_id='amazon.titan-embed-text-v1'),
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_type="mmr", # Also test "similarity"
    search_kwargs={"k": 8},
)

system_prompt_template = """
You are a highly analytical code review assistant specializing in both security and functional review. 
Your task is to analyze source code and provide detailed insights through a multi-step reflection process.

Follow these steps for each analysis:

1. Initial Analysis:
   - First, analyze the provided context thoroughly
   - Form initial observations about the codebase
   - Note any areas where you need more information

2. Reflection:
   - Critically evaluate your initial observations
   - Identify any potential gaps or assumptions in your analysis
   - Consider security implications you might have missed
   - Think about how different components interact

3. Final Analysis:
   - Combine your initial analysis with your reflections
   - Prioritize findings based on importance
   - Provide concrete examples where relevant
   - Highlight any remaining uncertainties

Context for analysis:
{context}

Remember to:
- Support all claims with evidence from the code
- Highlight any assumptions you're making
- Identify areas where more investigation might be needed
- Consider both security and functionality aspects
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    ("human", """
Please analyze the following aspects of the codebase, following the reflection process outlined above:

{question}

Format your response in the following structure:
1. Initial Analysis
2. Reflection on Initial Findings
3. Final Comprehensive Analysis
""")
])

llm = ChatBedrock(
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
    model_kwargs={"temperature": 0.6},
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

user_question = """Tell me the following information about the code base I am providing you:
 - Purpose of the application
 - Web technologies used in the application
 - Templating language used in the application
 - Database used in the application
 - Authentication mechanisms used in the application
 - Authorization mechanisms used in the application
List libraries by their name, purpose, and version that are
used in the application for the following categories:
    - Security
    - Testing
    - Documentation
    - Build
    - Database
    - Authentication / Authorization
    - HTML Templating (ex: pug, handlebars)
    - CSS Frameworks (ex: bootstrap, tailwind)
    - widgets / UI components
"""
    
# This is an optional addition to stream the output in chunks
# for a chat-like experience
for chunk in chain.stream(user_question):
    print(chunk, end="", flush=True)