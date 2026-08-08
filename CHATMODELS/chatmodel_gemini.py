# Import LangChain's integration for interacting with Google's Gemini models
from langchain_google_genai import ChatGoogleGenerativeAI

# Import utility to load configuration values from environment variables
from dotenv import load_dotenv

# Load environment variables from the .env file.
# The Google API key is typically stored here instead of hardcoding
# credentials directly in the source code.
load_dotenv()


# Initialize the Gemini chat model.
# The model parameter determines which Gemini model will process the request.
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")


# Invoke the model with a natural-language prompt.
# invoke() performs a synchronous request and returns an AIMessage
# containing the model's response and additional metadata.
result = model.invoke("What is the capital of USA")


# Extract and display only the textual content from the AIMessage.
print(result.content)
