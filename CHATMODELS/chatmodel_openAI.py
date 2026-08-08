# Import LangChain's integration for OpenAI chat models.
from langchain_openai import ChatOpenAI

# Utility for loading configuration and API credentials
# from environment variables.
from dotenv import load_dotenv

# Load environment variables from the .env file.
# The OpenAI API key should be stored here rather than hardcoded.
load_dotenv()


# Initialize the OpenAI chat model.
# temperature controls response randomness/creativity.
# max_completion_tokens limits the maximum number of tokens
# the model can generate.
model = ChatOpenAI(model="gpt-4", temperature=1.3, max_completion_tokens=10)


# Invoke the model with a natural-language prompt.
# The response is returned as an AIMessage object.
result = model.invoke("generate a poem")


# Extract and print the textual content from the AIMessage.
print(result.content)
