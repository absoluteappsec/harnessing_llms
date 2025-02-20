from langchain.agents import create_react_agent
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from typing import Optional, Type
from langchain.callbacks.manager import CallbackManagerForToolRun
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")

class CustomSearchTool(BaseTool):
    name: str = "custom_search"
    description: str = "Useful for when you need to answer questions about code"
    args_schema: Type[SearchInput] = SearchInput

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        faiss_db_path = "../vector_databases/vtm_faiss"
        db = FAISS.load_local(
            faiss_db_path, 
            BedrockEmbeddings(model_id='amazon.titan-embed-text-v1'),
            allow_dangerous_deserialization=True
        )
        return db.similarity_search(query)

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        raise NotImplementedError("custom_search does not support async")

# Define tools and LLM
tools = [CustomSearchTool()]
llm = ChatBedrock(
    model_id='anthropic.claude-3-5-haiku-20241022-v1:0',
    model_kwargs={"temperature": 0.6},
)

# Define instructions and prompt
instructions = """
You are an agent designed to detect Insecure Direct Object Reference (IDOR) vulnerabilities in Python code, specifically Django applications.

### **What is an IDOR?**
IDOR vulnerabilities occur when an application retrieves or modifies a database record using user-supplied input (e.g., a record ID) without ensuring the user is authorized to access or modify that specific record. 

### **Clarifications**
1. A generic authorization check (e.g., checking if the user has general permissions to perform a type of action) is **NOT sufficient** to prevent IDOR.
2. Authorization must be **scoped to the specific record** being accessed or modified. For example, if the code is updating a `User` record, the authorization must validate that the user is allowed to modify that specific `user_id`.
3. If the user-supplied input (e.g., `user_id`) is not validated against the current user’s permissions or roles for the corresponding record, the code is insecure.

### **When to Flag Code as Insecure**
You must flag the code as insecure (`is_insecure: true`) if:
- The authorization function is unrelated to the type of database record being accessed or modified (e.g., authorizing based on a `Task` model but modifying a `User` record).
- The code does not validate whether the user has permissions for the **specific record** identified by user-supplied input (e.g., `user_id`).
- You are not 100% confident that the authorization prevents IDOR.

### **When to Flag Code as Secure**
You may only flag the code as secure (`is_insecure: false`) if:
1. The code uses a proper authorization decorator or function that ensures the user is allowed to access or modify the specific database record.
2. You are certain that the authorization mechanism is correctly applied to the record type being accessed.

IMPORTANT
---------
- A general authorization check (e.g., permissions to perform a general action like `can_create_project`) does not prevent IDOR unless it is tied to the specific database record being accessed or modified (e.g., the `user_id` in this case).
- Always verify whether the authorization mechanism explicitly validates the record against the current user’s permissions.
- If the authorization function is authorizing a user on something like a project but the IDOR exists because the user is looking up a User record, then its not enforcing authorzation AND IS INSECURE

### **TOOLS**
You have access to a vector database to search for code-related information. When looking up custom functions, ensure an **exact match** on the function name and carefully review its implementation to determine whether it ensures record-level authorization.

### **Output Format**
Your final response must be in JSON format, containing the following fields:
- `is_insecure`: (bool) Whether the code is considered insecure.
- `reason`: (str) The reason the code is considered insecure or secure.

### **Examples**

#### Example 1: Insecure Code
```python
@login_required
def update_user(request):
    user_id = request.GET.get('user_id')
    User.objects.filter(id=user_id).update(is_active=False)

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, 
or if you do not need to use a tool, 
you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Your Final Answer should be in JSON format 
with the following fields:

- is_insecure: (bool) whether the code is considered insecure
- reason: (str) the reason the code is considered insecure

Begin!

New input: {input}
{agent_scratchpad}
"""
prompt = PromptTemplate.from_template(instructions)

# Create agent and executor
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)

def analyze_code(input_code: str) -> dict:
    """
    Analyze the given code using the agent_executor and return the result.
    """
    response = agent_executor.invoke({"input": input_code})
    return response

if __name__ == "__main__":
    # Example input
    input_code = """
    @login_required
    @user_passes_test(can_create_project)
    def update_user_active(request):
        user_id = request.GET.get('user_id')
        User.objects.filter(id=user_id).update(is_active=False)
    """
    result = analyze_code(input_code)
    print(result)
