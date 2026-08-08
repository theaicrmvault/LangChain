# Import LangChain's OpenAI integration for text-completion models.
from langchain_openai import OpenAI

# Utility for loading configuration values from environment variables.
from dotenv import load_dotenv

# Load environment variables from the .env file.
# The OpenAI API key should be stored here rather than hardcoded
# in the source code.
load_dotenv()


# Initialize the OpenAI text-completion model.
# Unlike ChatOpenAI, this interface is designed for traditional
# text-completion models that receive a text prompt directly.
llm = OpenAI(model="gpt-3.5-turbo-instruct")


# Send a text prompt to the completion model.
# invoke() executes the model request and returns the generated text.
result = llm.invoke("What is the capital of India")


# Print the generated text returned by the completion model.
print(result)
