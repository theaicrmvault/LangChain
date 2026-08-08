# Import ChatOpenAI to interact with an OpenAI chat model
from langchain_openai import ChatOpenAI

# Import message types used to maintain the conversation history
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# Import load_dotenv to load environment variables from the .env file
from dotenv import load_dotenv

# Load environment variables such as the OPENAI_API_KEY
load_dotenv()


# Create the ChatOpenAI model
model = ChatOpenAI()


# Create a list to store the complete conversation history
history = []


# Add the system instruction at the beginning of the conversation
history.append(SystemMessage(content="You are a helpful assistant."))


# Continuously accept user input until the user types "exit"
while True:

    # Get the user's message
    input_text = input("You: ")

    # Stop the conversation if the user enters "exit"
    if input_text.lower() == "exit":
        break

    # Add the user's message to the conversation history
    history.append(HumanMessage(content=input_text))

    # Send the complete conversation history to the model
    # The model uses previous messages to maintain context
    result = model.invoke(history)

    # Display the AI's response
    print("AI: " + result.content)

    # Add the AI's response to the conversation history
    # This allows the model to remember its previous response
    # in the next iteration
    history.append(AIMessage(content=result.content))
