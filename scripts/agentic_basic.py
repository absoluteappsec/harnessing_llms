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

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        faiss_db_path = "../vector_databases/vtm_faiss"
        db = FAISS.load_local(
            faiss_db_path,
            BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0"),
            allow_dangerous_deserialization=True,
        )
        return db.similarity_search(query)

    async def _arun(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        raise NotImplementedError("custom_search does not support async")


# Define tools and LLM
tools = [CustomSearchTool()]
llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    model_kwargs={"temperature": 0.6},
)

# Define instructions and prompt
instructions = """
You are an agent designed to analyze Python code for potential Insecure Direct Object Reference (IDOR) vulnerabilities.

### Analysis Process
1. Initial Review:
   - Identify where the code accesses or modifies database records
   - Locate user-supplied input that influences record access
   - Find authorization checks in the code

2. Reflection Questions:
   Consider these questions carefully:
   - How does the code determine which records a user can access?
   - What prevents a user from accessing records belonging to others?
   - Is there a mismatch between authorization scope and data access?
   - Could changing the input parameters bypass the authorization?

3. Challenge Initial Assessment:
   - What assumptions did you make about the authorization?
   - Are you certain the authorization check applies to the specific record?
   - What would an attacker try first to bypass these controls?

### **TOOLS**
You have access to a vector database to search for code-related information. Use it to understand how custom functions handle authorization.

### **Output Format**
Your final response must be in JSON format, containing the following fields:
- `is_insecure`: (bool) Whether the code is considered insecure.
- `reason`: (str) The reason the code is considered insecure or secure.

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
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
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
