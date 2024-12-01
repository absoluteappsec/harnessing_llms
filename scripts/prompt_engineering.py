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
You are a helpful code review assistant who is proficient
in both security as well as functional review. You will be
provided source code of a web application and tasked with
answering questions about it.

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
    model_id='anthropic.claude-3-haiku-20240307-v1:0',
    model_kwargs={"temperature": 0.6},
)



chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# This is an optional addition to stream the output in chunks
# for a chat-like experience
for chunk in chain.stream(
    """Tell me the following information about the code base I am providing you:
     - Purpose of the application
     - Web technologies used in the application
     - Templating language used in the application
     - Database used in the application
     - Authentication mechanisms used in the application
     - Authorization mechanisms used in the application


    List libraries by their name, purpose, and version that are
    used in the vtm application for the following categories:
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
    ):
    print(chunk, end="", flush=True)