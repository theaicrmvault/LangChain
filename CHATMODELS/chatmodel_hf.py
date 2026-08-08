# Import LangChain integrations for Hugging Face hosted models.
# HuggingFaceEndpoint connects to a model running on Hugging Face's
# inference infrastructure rather than loading the model locally.
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Utility for loading configuration such as API credentials
# from environment variables.
from dotenv import load_dotenv

# Load environment variables from the .env file.
# Hugging Face authentication credentials should be stored here
# instead of being hardcoded in the source code.
load_dotenv()


# Configure the Hugging Face hosted model endpoint.
# repo_id identifies the model to use.
# task defines the type of inference operation requested.
llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct", task="text-generation")


# Wrap the endpoint with LangChain's standardized chat-model interface.
# This allows the application to interact with the model using
# the common LangChain invoke() abstraction.
chat_model = ChatHuggingFace(llm=llm)


# Send the prompt to the remotely hosted model.
# The request is sent to the Hugging Face inference service,
# and the response is returned as an AIMessage.
result = chat_model.invoke("generate a classic poem")


# Extract and display the generated text from the AIMessage.
print(result.content)
