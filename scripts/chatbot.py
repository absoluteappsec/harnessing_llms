from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

faiss_db_path = "../vector_databases/acmeco_sec_guide_faiss"
db = FAISS.load_local(
    faiss_db_path, 
    BedrockEmbeddings(model_id='amazon.titan-embed-text-v1'),
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8},
)

# Initialize the ChatBedrock LLM
llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    model_kwargs={"temperature": 0.1}
)

# Define the chat template
chat_template = """
You are an expert and helpful application security engineer.
Engage in a conversation with the user and provide clear 
and concise responses about software security. Use the following
context to answer the questions:

<context>
{context}
</context>

If you are not absolutely certain about a response which should only come
from the context, reply "I am sorry please reach out to the security team directly."

User: {question}
Assistant:
"""

prompt = PromptTemplate(template=chat_template)

chat_chain = (
     {
        "question": RunnablePassthrough(),
        "context": retriever,
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Command-line chat application
def chat():
    print("Chat Assistant (type 'exit' to quit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        try:
            # This is an optional addition to stream the output in chunks
            # for a chat-like experience
            for chunk in chat_chain.stream(user_input):
                print(chunk, end="", flush=True)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat()
