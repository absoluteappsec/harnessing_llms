from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, get_buffer_string
from typing import Dict, List

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

faiss_db_path = "../vector_databases/juice_shop.faiss"
db = FAISS.load_local(
    faiss_db_path,
    BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0"),
    allow_dangerous_deserialization=True,
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 30},
)

# Initialize the ChatBedrock LLM
llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    model_kwargs={"temperature": 0.1},
)

# Define the chat template with chat history
chat_template = """
Engage in a conversation with the user and provide clear 
and concise responses using the data provided in the context section.

Chat History:
{chat_history}

<context>
{context}
</context>

User: {question}
Assistant:
"""

prompt = PromptTemplate.from_template(chat_template)

# Define the main chat processing chain
chat_chain = (
    {
        "question": RunnablePassthrough(),
        "context": lambda x: retriever.invoke(x["question"] if isinstance(x, dict) else x),
        "chat_history": RunnablePassthrough(),  # Placeholder for history
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Create a custom message history store
class InMemoryChatMessageHistory(BaseChatMessageHistory):
    """In-memory implementation of chat message history."""
    
    def __init__(self):
        self.messages = []
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history."""
        self.messages.append(message)
    
    def clear(self) -> None:
        """Clear the message history."""
        self.messages = []

# Manage chat histories for multiple sessions
class ChatMessageHistoryManager:
    """Manages chat message history for multiple sessions."""
    
    def __init__(self):
        self.histories = {}
    
    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create a history instance for the session."""
        if session_id not in self.histories:
            self.histories[session_id] = InMemoryChatMessageHistory()
        return self.histories[session_id]

# Initialize the history manager
history_manager = ChatMessageHistoryManager()

# Function to get chat history for a session
def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    return history_manager.get_history(session_id)

# Wrap the chat chain with history
chat_chain_with_history = RunnableWithMessageHistory(
    chat_chain,
    get_chat_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)


# Command-line chat application
def chat():
    print("Juice Shop Assistant (type 'exit' to quit)")
    session_id = "default"  # Change if multi-session support is needed

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            # Get AI response
            response = ""
            for chunk in chat_chain_with_history.stream(
                {"question": user_input}, 
                config={"configurable": {"session_id": session_id}}
            ):
                response += chunk
                print(chunk, end="", flush=True)

        except EOFError:
            print("\nExiting due to EOF")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    chat()
