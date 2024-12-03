from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnableSequence, RunnablePassthrough
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser

# Load Env Variables
from dotenv import load_dotenv
load_dotenv()


# Define the LLM
llm = ChatBedrock(
    model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"temperature": 0.5},
)

# Define the first question prompt
first_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ]
)

# Define the second question prompt
second_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            """The response provided earlier was: {previous_response}.
            Using this response as a basis, compare the key benefits of Python against those of JavaScript. 
            Focus on highlighting both strengths and weaknesses of each language."""
        ),
    ]
)

# Combine into a multi-step chain
chain = RunnableSequence(
    RunnableMap({
        "llm_response": first_prompt | llm | StrOutputParser(),  # Generate response for the first prompt
    }),
    RunnableMap({
        "previous_response": lambda inputs: inputs["llm_response"],  # Map llm_response to previous_response
        "next_question": RunnablePassthrough(),  # Pass next_question directly
    }) | second_prompt | llm | StrOutputParser()  # Generate final response for the second prompt
)





# Inputs for the chain
inputs = {
    "question": "What are the primary benefits of using Python?",
    "next_question": f"Compare those key benefits against using JavaScript.",
}



for chunk in chain.stream(inputs):
    print(chunk, end="", flush=True)
