# Import OpenAI's embedding model integration from LangChain.
# Embeddings convert text into numerical vectors that capture
# the semantic meaning of the input.
from langchain_openai import OpenAIEmbeddings

# Utility for loading configuration values from environment variables.
from dotenv import load_dotenv

# Load environment variables from the .env file.
# The OpenAI API key should be stored here instead of hardcoding
# credentials in the source code.
load_dotenv()


# Initialize the OpenAI embedding model.
# text-embedding-3-small converts text into a semantic vector.
# dimensions=32 reduces the output vector to 32 dimensions.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)


# Convert the user's query into an embedding vector.
# embed_query() is specifically intended for generating a vector
# representation of a search/query input.
query = "Explain the theory of relativity in simple terms"

vector = embeddings.embed_query(query)


# Display the generated query embedding.
print(str(vector))
