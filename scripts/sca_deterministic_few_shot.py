import os
import git
from langchain_aws import ChatBedrock
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
import time


# Load Env Variables
from dotenv import load_dotenv
load_dotenv()

repo_url = 'https://github.com/redpointsec/vtm.git'
local_path = './repo'

if os.path.isdir(local_path) and os.path.isdir(os.path.join(local_path, '.git')):
    print("Directory already contains a git repository.")
else:
    try:
        repo = git.Repo.clone_from(repo_url, local_path)
        print(f"Repository cloned into: {local_path}")
    except Exception as e:
        print(f"An error occurred while cloning the repository: {e}")


# THIS LOADS ONLY PYTHON EXTENSION FILES, CHANGE AS NEEDED
python_files = {}
# Traverse the directory recursively
for root, _, files in os.walk(local_path):
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            try:
                # Read the contents of the Python file
                with open(file_path, 'r', encoding='utf-8') as f:
                    python_files[file_path] = f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

llm = ChatBedrock(
    model_id='us.anthropic.claude-3-5-haiku-20241022-v1:0',
    model_kwargs={"temperature": 0.2},
)

system_prompt_template = """
You are a helpful secure code review assistant who is given acess to a
code base stored in vector format. You will be asked questions about that code.
Please provide helpful and accurate responses to the best of your ability.

</context>
{context}
</context>

<background>
 Django's ORM methods automatically handle SQL escaping
 in order to prevent SQL Injection attacks. Unsafe SQL
 queries can only be run with the following functions:
 `.raw()`, `.execute()`, `.extra()`.
</background>
"""

examples = [

    {
        "context": """
            def get_users(username):
                cursor = connection.cursor()
                cursor.execute("SELECT * FROM users WHERE username = '%s'" % username)
                return cursor.fetchall()
        """,
        "question": "Identify everywhere in the Django application code base that contains SQL Injection vulnerablities.",
        "answer": "This code is vulnerable to SQL Injection because it takes the username (user input) and directly concatenates it into the SQL query."
    },
    {
        "context": """
            def get_users(username):
                cursor = connection.cursor()
                cursor.execute("SELECT * FROM users WHERE username = %s", [username])
                return cursor.fetchall()
        """,
        "question": "Identify everywhere in the Django application code base that contains SQL Injection vulnerablities.",
        "answer": """
            Although the code uses the `cursor.execute()` method, it is using
            parameterized queries, therefore it is not
            vulnerable to SQL Injection.
        """
    },
    {
        "context":"""
            if re.match('.*?(rm|sudo|wget|curl|su|shred) .*',ip,re.I):
                data = "Nice try on the dangerous commands, but no"
            else:
                cmd = "ping -c 5 %s" % ip
                data = subprocess.getoutput(cmd)
        """,
        "question":"Identify everywhere in the Django application code base that contains SQL Injection vulnerablities.",
        "answer": """
            This code is not vulnerable to SQL Injection because it is command injection so I will not mention it.
        """
    },
    {
        "context": """
            ssn = forms.CharField(max_length=11, required=False)
        """,
        "question": "Identify everywhere in the Django application code base that contains SQL Injection vulnerablities.",
        "answer": """
            This code is not vulnerable to SQL Injection because it is a form field definition.
        """
    }
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "<question>{question}</question>\n<context>{context}</context>"),
        ("ai", "{answer}"),
    ]
)


few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        few_shot_prompt,
        ("human", """<question>{question}</question>""")
    ]
)


question = """
Identify everywhere in the Django application code base that contains
SQL Injection vulnerablities. Only tell me about the code that is vulnerable
to SQL Injection. Don't mention code that is not vulnerable to SQL Injection.
"""

# Now we will iterate over the Python files and analyze them
# and store the results for later processing
for file_path, content in python_files.items():
    code = content
    # Retrieve filename for reference
    filename = file_path
    # Create a chain of operations to run the code through
    chain = (
        { "context": RunnablePassthrough() , "question": RunnablePassthrough()}
        | final_prompt
        | llm
        | StrOutputParser()
    )

    # This is an optional addition to stream the output in chunks
    # for a chat-like experience
    title = f"\n\nAnalyzing code from {filename}"
    print(title)
    print("=" * len(title))
    try:
        for chunk in chain.stream({"question": question,"context": code}):
            print(chunk, end="", flush=True)
    except Exception as e:
        time.sleep(30)
        continue






    
