from langchain.agents import create_react_agent
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type, List, Dict
from langchain.callbacks.manager import CallbackManagerForToolRun
from dotenv import load_dotenv
import os
import json

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
                for file in filenames:
                    if file.endswith(('.py', '.rb', '.js', '.php', '.html')):  # Focus on common web app files
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

# Phase 1: Framework Detection
framework_detection_prompt = """
You are an expert at identifying web application frameworks and technology stacks.
Your task is to examine the codebase and identify:
1. The primary web framework being used
2. Any significant libraries or dependencies
3. The overall architecture pattern

Examine the file structure, dependencies, and code patterns to make this determination.

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
Thought: I have completed my framework analysis
Final Answer: [your JSON response here]
```

Your final response must be a JSON object with this structure:
{{
    "framework": str,  // Primary web framework
    "dependencies": List[str],  // Key libraries and dependencies
    "architecture_pattern": str,  // e.g., "MVC", "REST API", etc.
    "key_files": List[str]  // Important framework-specific files
}}

Begin your framework analysis with the directory provided.

New input: {input}
{agent_scratchpad}
"""

# Phase 2: Critical File Analysis
critical_file_analysis_prompt = """
You are an expert security auditor focusing on identifying security-critical files.
Based on the framework analysis provided, examine the codebase for files that are most likely to contain security vulnerabilities.

Previous framework analysis: {framework_analysis}

Focus on these types of files:
1. Routes/URL handlers
2. Controllers/Views
3. Models/Data access
4. Configuration/Settings
5. Authentication/Authorization code
6. Input validation and sanitization
7. Template rendering
8. Utility functions

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
Thought: I have completed my critical file analysis
Final Answer: [your JSON response here]
```

Your final response must be a JSON object with this structure:
{{
    "critical_files": [
        {{
            "file": str,  // File path
            "category": str,  // e.g., "routes", "auth", "data_access"
            "risk_level": str,  // "HIGH", "MEDIUM", or "LOW"
            "reason": str  // Why this file is security-critical
        }}
    ]
}}

Begin your critical file analysis.

New input: {input}
{agent_scratchpad}
"""

# Phase 3: Vulnerability Assessment
vulnerability_assessment_prompt = """
You are an expert security auditor performing a detailed vulnerability assessment.
Based on the critical files identified, analyze each file for specific security vulnerabilities.

Previous analyses:
Framework Analysis: {framework_analysis}
Critical Files: {critical_files}

IMPORTANT: You MUST analyze ALL files listed in the Critical Files analysis before providing your final answer.

ANALYSIS PROCESS:
1. First, extract and list all files to analyze from the Critical Files analysis
2. For each file:
   - View and analyze the file's contents
   - Document ALL vulnerabilities found using this format for each finding:
     ```
     File: [filename]
     Vulnerability Type: [type]
     Location: [line numbers]
     Code: [vulnerable code]
     Severity: [HIGH/MEDIUM/LOW]
     Description: [detailed description]
     Recommendation: [specific fix]
     Related Files: [any related files]
     ```
   - After documenting each vulnerability, explicitly state "Completed analysis of [filename]"
3. After analyzing ALL files, combine ALL findings into the final JSON response

For each file, check for these vulnerability categories:
1. Critical Risk Application Security flaws such as:
  - Mass Assignment
  - No/SQL Injection
  - Remote Code Execution
  - Command Injection
  - Insecure Direct Object Reference / Broken Object Level Access
  - Authentication Security weaknesses
  - SSRF
  - Logic Flaws
  - Weak Cryptography
  - Cross-Site Request Forgery
  - Server Side Template Injection
  - Cross-Site Scripting
  - Insecure hashing algorithms or generally insecure cryptography practices
  - Directory or File traversal / Remote File Inclusion / Local File Inclusion

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

You MUST use this exact process:

```
Thought: Listing all files to analyze from Critical Files
Files: [list all files]

[For each file:]
Thought: Now analyzing [current file]
Action: view_file
Observation: [file contents]
[If vulnerabilities found:]
Vulnerability Found:
File: [filename]
Vulnerability Type: [type]
Location: [line numbers]
Code: [vulnerable code]
Severity: [HIGH/MEDIUM/LOW]
Description: [detailed description]
Recommendation: [specific fix]
Related Files: [any related files]
[Repeat for each vulnerability found in this file]
Thought: Completed analysis of [current file]

[After ALL files are analyzed:]
Thought: I have completed analysis of all files. Combining all findings into final response.
Final Answer: [JSON with ALL findings]
```

Your final response must be a JSON object with this structure:
{{
    "vulnerabilities": [
        {{
            "type": str,  // Category of vulnerability
            "vulnerable_code": str,  // Code that contains the vulnerability
            "file": str,  // File path
            "line_numbers": List[int],  // Line numbers of vulnerable code
            "severity": str,  // "HIGH", "MEDIUM", or "LOW"
            "description": str,  // Detailed description
            "recommendation": str,  // Specific fix recommendation
            "related_files": List[str]  // Other files involved
        }}
    ],
    "files_analyzed": List[str],  // List of all files that were analyzed
    "vulnerability_summary": {{
        "total_vulnerabilities": int,
        "vulnerabilities_by_file": {{
            "filename": List[str]  // List of vulnerability types found in each file
        }}
    }}
}}

CRITICAL REMINDERS:
1. Document EACH vulnerability as you find it using the exact format above
2. Do not move to the next file until you have documented ALL vulnerabilities in the current file
3. Your final JSON must include ALL vulnerabilities from ALL files
4. If a file has no vulnerabilities, still mark it as analyzed

Begin your vulnerability assessment.

New input: {input}
{agent_scratchpad}
"""

def create_agent_executor(prompt_template: str) -> AgentExecutor:
    """Create an agent executor with the given prompt template."""
    prompt = PromptTemplate.from_template(prompt_template)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

def analyze_security(directory_path: str) -> dict:
    """
    Perform a phased security analysis of the given directory.
    """
    # Phase 1: Framework Detection
    framework_executor = create_agent_executor(framework_detection_prompt)
    framework_analysis = framework_executor.invoke({"input": directory_path})
    
    # Phase 2: Critical File Analysis
    file_analysis_prompt = critical_file_analysis_prompt
    critical_file_executor = create_agent_executor(file_analysis_prompt)
    critical_files = critical_file_executor.invoke({
        "input": directory_path,
        "framework_analysis": json.dumps(framework_analysis, indent=2)
    })
    
    # Phase 3: Vulnerability Assessment
    vuln_assessment_prompt = vulnerability_assessment_prompt
    vulnerability_executor = create_agent_executor(vuln_assessment_prompt)
    vulnerabilities = vulnerability_executor.invoke({
        "input": directory_path,
        "framework_analysis": json.dumps(framework_analysis, indent=2),
        "critical_files": json.dumps(critical_files, indent=2)
    })
    
    # Combine all results
    return {
        "framework_analysis": framework_analysis,
        "critical_files": critical_files,
        "vulnerabilities": vulnerabilities
    }

if __name__ == "__main__":
    # Get repository path from environment variable or use default
    repo_path = os.getenv("REPO_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repo"))
    
    # Ensure the path exists
    if not os.path.exists(repo_path):
        print(f"Error: Repository path '{repo_path}' does not exist")
        exit(1)
    
    # Run the phased analysis
    result = analyze_security(repo_path)
    
    # Print results in a structured format
    print("\n=== Framework Analysis ===")
    print(json.dumps(result["framework_analysis"], indent=2))
    
    print("\n=== Critical Files ===")
    print(json.dumps(result["critical_files"], indent=2))
    
    print("\n=== Vulnerabilities ===")
    print(json.dumps(result["vulnerabilities"], indent=2))
