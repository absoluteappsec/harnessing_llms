from langchain.agents import create_react_agent
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type, List
from langchain.callbacks.manager import CallbackManagerForToolRun
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class ListFilesInput(BaseModel):
    directory: str = Field(description="Directory path to list contents from")

class ViewFileInput(BaseModel):
    filepath: str = Field(description="Path to the file to view")

class ListFilesTool(BaseTool):
    name: str = "list_files"
    description: str = "Lists files and directories in the specified directory"
    args_schema: Type[ListFilesInput] = ListFilesInput

    def _run(self, directory: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            files = []
            for root, dirs, filenames in os.walk(directory):
                for dir in dirs:
                    files.append(f"Directory: {os.path.join(root, dir)}")
                for file in filenames:
                    if file.endswith(('.py', '.rb', '.js', '.php')):  # Focus on common web app files
                        files.append(f"File: {os.path.join(root, file)}")
            return "\n".join(files)
        except Exception as e:
            return f"Error listing directory: {str(e)}"

class ViewFileTool(BaseTool):
    name: str = "view_file"
    description: str = "Views the contents of a specified file"
    args_schema: Type[ViewFileInput] = ViewFileInput

    def _run(self, filepath: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            with open(filepath, 'r') as file:
                return file.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

# Define tools and LLM
tools = [ListFilesTool(), ViewFileTool()]
llm = ChatBedrock(
    model_id='us.anthropic.claude-3-5-haiku-20241022-v1:0',
    model_kwargs={"temperature": 0.6},
)

# Define instructions and prompt
instructions = """
You are an expert security auditor tasked with analyzing code for common web application vulnerabilities.
Your goal is to thoroughly examine the codebase for the following security issues:

1. SQL Injection
   - Look for raw SQL queries with user input
   - Check for proper use of parameterized queries or ORM methods
   - Identify unsafe string concatenation in queries

2. Cross-Site Scripting (XSS)
   - Check for unescaped user input in HTML/JavaScript output
   - Look for proper use of template escape functions
   - Identify unsafe innerHTML or document.write usage

3. Cross-Site Request Forgery (CSRF)
   - Check for CSRF token validation
   - Look for proper middleware usage
   - Identify forms without CSRF protection

4. Mass Assignment
   - Look for bulk updates or creates with user input
   - Check for proper attribute filtering
   - Identify unprotected model attributes

5. Command Injection
   - Look for shell command execution
   - Check for proper input sanitization
   - Identify unsafe use of eval() or similar functions

6. Server-Side Request Forgery (SSRF)
   - Look for URL fetching with user input
   - Check for proper URL validation
   - Identify unsafe HTTP client usage

### Analysis Process
1. First, use the list_files tool to discover relevant code files
2. For each relevant file:
   - Use view_file to examine its contents
   - Analyze the code for each vulnerability type
   - Document any findings with specific line numbers and explanations

### Output Format
Your final response must be a JSON object with the following structure:
{{
    "vulnerabilities": [
        {{
            "type": str,  // One of: "SQL_INJECTION", "XSS", "CSRF", "MASS_ASSIGNMENT", "COMMAND_INJECTION", "SSRF"
            "file": str,  // File path where the vulnerability was found
            "line_numbers": List[int],  // Line numbers of the vulnerable code
            "severity": str,  // One of: "HIGH", "MEDIUM", "LOW"
            "description": str,  // Detailed description of the vulnerability
            "recommendation": str  // Specific fix recommendation
        }}
    ]
}}

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

When you have completed your analysis, you MUST use the format:

```
Thought: I have completed my security analysis
Final Answer: [your JSON response here]
```

Begin your security audit with the directory provided in the input.

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

def analyze_security(directory_path: str) -> dict:
    """
    Analyze the given directory for security vulnerabilities.
    """
    response = agent_executor.invoke({"input": directory_path})
    return response

if __name__ == "__main__":
    # Example usage
    directory_to_scan = "./repo"  # Update this to your target directory
    result = analyze_security(directory_to_scan)
    print(result)
