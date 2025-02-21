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
You are an expert security auditor tasked with analyzing code for web application vulnerabilities.
Follow this systematic approach for your security analysis:

1. Framework Detection Phase
   - First examine the codebase to identify what web framework is being used
   - Look for framework-specific patterns, file structures, and dependencies
   - Use your knowledge of web frameworks to recognize common patterns

2. Critical File Analysis Phase
   - Based on the identified framework, locate the key files where security issues commonly occur:
     * Routes/URL handlers
     * Controllers/Views
     * Models/Data access
     * Configuration/Settings
     * Authentication/Authorization code
     * Input validation and sanitization
     * Template rendering
     * Utility functions
   - Use your knowledge of the framework's conventions to find these files

3. Vulnerability Assessment Phase
   For each critical file, analyze for these vulnerability categories:

   a) Authentication & Authorization
      - Improper access controls
      - Authentication bypasses
      - Session management issues

   b) Input Validation & Sanitization
      - SQL Injection
      - Cross-Site Scripting (XSS)
      - Command Injection
      - Path Traversal

   c) Configuration & Information Exposure
      - Security misconfigurations
      - Sensitive data exposure
      - Insecure defaults

   d) Framework-Specific Issues
      - Known framework vulnerabilities
      - Misuse of framework features
      - Insecure implementations

4. Deep Analysis Phase
   - When you find a potential vulnerability, recursively analyze related files
   - Follow data flows to validate the issue
   - Check for any mitigating controls

### Output Format
Your final response must be a JSON object with the following structure:
{{
    "framework_detected": str,  // The identified web framework
    "critical_files": [
        {{
            "file": str,  // File path
            "purpose": str,  // File's role (e.g., "routes", "controller", "model")
            "security_impact": str  // Why this file is security-critical
        }}
    ],
    "vulnerabilities": [
        {{
            "type": str,  // Category of vulnerability
            "file": str,  // File path where found
            "line_numbers": List[int],  // Line numbers of vulnerable code
            "severity": str,  // "HIGH", "MEDIUM", or "LOW"
            "description": str,  // Detailed description
            "recommendation": str,  // Specific fix recommendation
            "related_files": List[str]  // Other files involved
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