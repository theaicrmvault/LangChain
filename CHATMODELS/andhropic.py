# Import the Anthropic chat model integration from LangChain
from langchain_anthropic import ChatAnthropic

# Import utility to load environment variables from the .env file
from dotenv import load_dotenv

# Load environment variables such as the Anthropic API key
# stored in the .env file into the application's environment
load_dotenv()


# Initialize the Claude chat model
# temperature controls the randomness/creativity of the response
# max_completion_tokens limits the maximum number of tokens generated
model = ChatAnthropic(model="claude-2", temperature=1.3, max_completion_tokens=10)

# Send a prompt to the model and wait synchronously for the response
result = model.invoke("generate poem")
