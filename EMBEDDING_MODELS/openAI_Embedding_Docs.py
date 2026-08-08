# Import OpenAI's embedding model integration from LangChain.
# Embeddings convert text into numerical vectors that capture
# the semantic meaning of the text.
from langchain_openai import OpenAIEmbeddings

# Utility for loading configuration values from environment variables.
from dotenv import load_dotenv

# Load environment variables from the .env file.
# The OpenAI API key should be stored here instead of hardcoding
# credentials in the source code.
load_dotenv()


# The document has been manually divided into smaller text chunks.
# In a production RAG system, chunking is typically performed
# automatically based on chunk size and document structure.
documents = [
    "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space",
    "and time. It consists of two main parts: special relativity and general relativity.",
    "Special relativity, published in 1905, introduced the idea that the laws of physics are the same for all non-accelerating observers and that the speed of light is constant regardless of the motion of the light source.",
    "This led to the famous equation E=mc^2, which shows the relationship between energy and mass.",
    "General relativity, published in 1915, expanded on this by describing how gravity is not just a force between masses but a curvature of spacetime caused by mass and energy.",
    "This theory has been confirmed by many experiments and observations, such as the bending of light",
]


# Initialize the OpenAI embedding model.
# text-embedding-3-small converts each text chunk into a
# numerical vector representing its semantic meaning.
#
# dimensions=32 reduces the default embedding size to 32 dimensions.
# Smaller vectors reduce storage and processing costs but may
# reduce semantic representation quality.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)


# Generate one embedding vector for each document chunk.
# The output is a list of vectors where:
#
# number of vectors = number of documents
# vector size = 32 dimensions
vector = embeddings.embed_documents(documents)


# Display the generated embedding vectors.
print(str(vector))
