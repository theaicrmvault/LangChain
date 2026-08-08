# Import ChatPromptTemplate to create a structured chat prompt
# Import MessagesPlaceholder to dynamically insert messages into the prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create a reusable chat prompt template
template = ChatPromptTemplate(
    [
        # Defines the behavior/role of the AI assistant
        ("system", "You are a helpful assistant."),
        # Dynamically inserts previously stored chat messages
        # The value will be provided using the "stored_messages" key
        MessagesPlaceholder(variable_name="stored_messages"),
        # Human message containing a dynamic input variable
        ("human", "tell me about {input_text}"),
    ]
)


# Create an empty list to store previous conversation messages
chat_history = []


# Read previously stored messages from a text file
with open("stored_messages.txt") as f:
    # Add each line from the file to the chat history
    chat_history.extend(f.readlines())


# Fill the template variables:
# - input_text → current user question
# - stored_messages → previous conversation history
result = template.invoke(
    {
        "input_text": "Tell me example again",
        "stored_messages": chat_history,
    }
)


# Print the final prompt containing:
# System message + stored messages + current human message
print(result)
