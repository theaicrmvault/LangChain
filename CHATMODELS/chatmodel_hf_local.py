# Import LangChain integrations for Hugging Face models.
# HuggingFacePipeline runs the model locally through a Transformers pipeline,
# while ChatHuggingFace provides a chat-model interface on top of it.
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Utility for loading configuration from environment variables.
from dotenv import load_dotenv

# Load environment variables from the .env file.
# Useful when additional configuration or credentials are required.
load_dotenv()


# Create a Hugging Face text-generation pipeline from a pretrained model.
# The model is downloaded and executed locally rather than calling a hosted
# LLM API.
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    # Configure generation behavior for the underlying model.
    # Higher temperature generally produces more varied responses.
    pipeline_kwargs={"temperature": 1.3},
)


# Wrap the text-generation pipeline with LangChain's chat-model interface.
# This allows the model to be used with the standard LangChain
# model.invoke() abstraction.
model = ChatHuggingFace(llm=llm)


# Send the user prompt to the local chat model.
# invoke() returns an AIMessage containing the generated response.
result = model.invoke("Explain the theory of relativity in simple terms")


# Extract and print only the generated text from the AIMessage.
print(result.content)
