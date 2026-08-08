# Import ChatPromptTemplate to create a structured chat prompt
from langchain_core.prompts import ChatPromptTemplate

# Create a chat prompt template with two messages:
# 1. System message → defines the assistant's behavior
# 2. Human message → contains a dynamic variable {input_text}
template = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant."),
        ("human", "tell me about {input_text}"),
    ]
)


# Fill the {input_text} variable with the actual value
# and generate the final prompt messages
result = template.invoke({"input_text": "Prompt Caching"})


# Print the generated prompt structure
print(result)
