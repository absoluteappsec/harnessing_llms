from langchain.agents import create_react_agent
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.callbacks.manager import CallbackManagerForToolRun
from dotenv import load_dotenv
import os
import httpx

# Load environment variables
load_dotenv()


class HttpInput(BaseModel):
    url: str = Field(description="a url to make a request to")
    method: str = Field(description="the http method to use (GET or POST)", default="GET")
    data: Optional[dict] = Field(description="the data to send with a POST request", default=None)


class HttpTool(BaseTool):
    name: str = "http_tool"
    description: str = "Useful for when you need to make a request to a url. Can be used for GET and POST requests."
    args_schema: Type[HttpInput] = HttpInput

    def _run(
        self, url: str, method: str = "GET", data: Optional[dict] = None, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        if method.upper() == "POST":
            response = httpx.post(url, data=data)
        else:
            response = httpx.get(url)
        headers = str(response.headers)
        body = response.text
        return f"Headers:\n{headers}\n\nBody:\n{body}"

    async def _arun(
        self, url: str, method: str = "GET", data: Optional[dict] = None, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        raise NotImplementedError("http_tool does not support async")


# Define tools and LLM
tools = [HttpTool()]
llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    model_kwargs={"temperature": 0.6},
)

# Define instructions and prompt
instructions = """
You are an agent designed to make an http request to a provided url and analyze the response using a multi-step reasoning process.

### **Analysis Process**
1. **Initial Request**: Make an HTTP request to the provided URL using the specified method (GET or POST).
2. **Response Analysis**: Analyze the response headers and body for the following information:
   - Status Code: (int) The HTTP status code of the response
   - Headers: (str) The headers of the response
   - Body: (str) The body of the response  
   - Security Considerations: (str) Any security considerations based on the response content
   - URLs: (list) Any URLs found in the response body
3. **Final Response**: Return the relevant information from the HTTP request in the following format:

### **Response Format**
- Status Code: (int) The HTTP status code of the response
- Headers: (str) The headers of the response
- Body: (str) The body of the response
- Security Considerations: (str) Any security considerations based on the response content
- URLs: (list) Any URLs found in the response body

### **TOOLS**
You have access to a tool that can make an http request to a provided url. It can handle both GET and POST requests.

### **Output Format**
Your final response must be the full response from the http request.

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


def run_agent(url: str) -> dict:
    """
    Analyze the given code using the agent_executor and return the result.
    """
    response = agent_executor.invoke({"input": url})
    return response


if __name__ == "__main__":
    # Example input for GET request
    url = "https://vtm.rdpt.dev/taskManager/login/?next=/"
    result = run_agent(url)
    print(result)

    # Example input for POST request
    # To run this, you would need to modify how the agent is invoked to handle structured input for the tool.
    # For example:
    # post_input = {
    #     "input": {
    #         "tool_input": {
    #             "url": "http://httpbin.org/post",
    #             "method": "POST",
    #             "data": {"key": "value"}
    #         }
    #     }
    # }
    # result = agent_executor.invoke(post_input)
    # print(result)
