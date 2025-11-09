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
    BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0"),
    allow_dangerous_deserialization=True,
)

retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 100},
)

system_prompt_template = """
Analyze source code and provide detailed security and functional insights as requested.

Context for analysis:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        (
            "human",
            """
Answer questions about the codebase:

{question}

""",
        ),
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

user_question = """Tell me about the application, its functionality, and any potential security issues you can identify from the codebase provided.
"""

# This is an optional addition to stream the output in chunks
# for a chat-like experience
for chunk in chain.stream(user_question):
    print(chunk, end="", flush=True)
