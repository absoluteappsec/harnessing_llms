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
You are a highly skilled and detail-oriented code review assistant with expertise in both application security and functional code analysis. Your role is to assist developers and security professionals by providing accurate, concise, and actionable insights.

Your task is to analyze the provided source code of a web application and answer specific questions about its functionality, security, and technologies. Always maintain a professional tone and prioritize clarity in your responses.

In your analysis:
- Clearly identify and explain the purpose and technologies used in the codebase.
- Highlight critical security mechanisms such as authentication and authorization.
- Provide details on libraries, tools, and frameworks, organized by their categories and roles in the application.
- When relevant, make recommendations for improving security or functionality.

Search the following codebase to answer the questions:

<code>
{code}
</code>

Use the following context to help answer questions:
<context>
{context}
</context>
"""

# CORRECT/FORMAL WAY TO PERFORM PROMPTING
prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_template),
                ("human", """<question>{question}</question>""")
            ]
)

# UNCOMMENT FOR OLLAMA/LLAMA
#llm = Ollama(model="llama3.1", temperature=0.6)

llm = ChatBedrock(
    model_id='us.anthropic.claude-3-5-haiku-20241022-v1:0',
    model_kwargs={"temperature": 0.6},
)


knowledge_base_file_path = "../data/juice_shop_knowledgebase.md"
with open(knowledge_base_file_path, 'r', encoding='utf-8') as file:
    context = file.read()

chain = (
     {
        "code": retriever,
        "question": RunnablePassthrough(),
        "context": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

user_question = """
Please analyze the codebase and create a comprehensive threat model by addressing the following:

1. **Assets and Trust Boundaries**:
   - What are the key assets (data, functionality) that need protection?
   - What are the trust boundaries in the application?
   - What sensitive data flows exist between components?

2. **Threat Actors and Attack Surfaces**:
   - Who are the potential threat actors targeting this application?
   - What are the primary attack surfaces exposed by the application?
   - What entry points could attackers use to compromise the system?

3. **Vulnerabilities and Security Controls**:
   - What potential vulnerabilities exist in the current implementation?
   - What security controls are currently in place?
   - Are there any missing or inadequate security controls?

4. **Risk Assessment**:
   - What are the highest risk threats to the application?
   - What potential impact could successful attacks have?
   - What is the likelihood of different types of attacks?

5. **Security Recommendations**:
   - What additional security controls should be implemented?
   - What architectural changes could improve security?
   - What security testing should be prioritized?
"""

for chunk in chain.stream(user_question, {"context":context}):
                print(chunk, end="", flush=True)