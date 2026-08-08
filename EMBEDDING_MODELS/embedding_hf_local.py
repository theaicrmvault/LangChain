# Import LangChain's Hugging Face embedding integration.
# HuggingFaceEmbeddings allows us to generate embeddings locally
# using a Sentence Transformers model.
from langchain_huggingface import HuggingFaceEmbeddings

# Sample text split into multiple chunks.
# Each chunk will be converted into a numerical vector independently.
text = [
    "The theory of relativity, developed by Albert Einstein ",
    "revolutionized our understanding of space and time. It consists of two main parts: special relativity and general relativity. Special relativity, published in 1905, introduced the idea that the laws of physics are the same for all non-accelerating observers",
    "and that the speed of light is constant regardless of the motion of the light source. This led to the famous equation E=mc^2, which shows the relationship ",
    "between energy and mass. General relativity, published in 1915, expanded on this by describing how gravity is not just a force between masses but a curvature of spacetime caused by mass and energy. This theory has been confirmed by many experiments and observations, such as the bending of light.",
]


# Initialize the embedding model.
# all-MiniLM-L6-v2 is a lightweight Sentence Transformers model
# that generates semantic vector representations locally.
#
# Unlike OpenAIEmbeddings, this approach does not require calling
# an external embedding API for inference.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Convert each text chunk into an embedding vector.
# The result is a list of numerical vectors, with one vector
# corresponding to each text chunk.
vector = embeddings.embed_documents(text)


# Display the generated embedding vectors.
print(str(vector))
