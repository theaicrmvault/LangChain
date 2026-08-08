# OpenAIEmbeddings converts text into numerical vector representations
# that capture the semantic meaning of the text.
from langchain_openai import OpenAIEmbeddings

# Utility for loading API credentials from environment variables.
from dotenv import load_dotenv

# Used to calculate cosine similarity between the query vector
# and document vectors.
from sklearn.metrics.pairwise import cosine_similarity

# NumPy is used for numerical/vector operations.
import numpy as np

# Load environment variables such as the OpenAI API key
# from the .env file.
load_dotenv()


# Initialize the embedding model.
# text-embedding-3-large generates high-dimensional semantic vectors.
# dimensions=300 reduces the output vector size to 300 dimensions,
# which can reduce storage and similarity-search costs.
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)


# Sample knowledge base containing documents that will be searched.
documents = [
    "Artificial intelligence is transforming healthcare by helping doctors diagnose diseases faster.",
    "Machine learning models require high-quality training data to produce accurate predictions.",
    "Salesforce Agentforce enables organizations to build AI-powered agents for customer support and automation.",
    "The Eiffel Tower is one of the most famous landmarks in Paris, France.",
    "Python is a popular programming language for AI, data science, and automation.",
    "Large Language Models can answer questions, summarize documents, and generate code.",
    "Cricket is one of the most popular sports in India.",
    "Virat Kohli scored another century in the international cricket match.",
    "Cats are independent pets that enjoy sleeping and climbing.",
    "Dogs are loyal companions and often enjoy playing fetch with their owners.",
    "A customer wants to reset their password because they cannot log in to the application.",
    "The user forgot their password and needs help accessing their account.",
    "The restaurant serves authentic Italian pizza, pasta, and tiramisu.",
    "The database connection timed out while processing the API request.",
    "My laptop battery drains quickly when running machine learning models locally.",
]


# Natural-language query that will be used to search the knowledge base.
query = "Tell me about the Eiffel Tower and its history."


# Convert every document into an embedding vector.
# Each document is represented as a numerical vector capturing
# its semantic meaning.
doc_embeddings = embeddings.embed_documents(documents)


# Convert the user's query into an embedding vector using the
# same embedding model used for the documents.
query_embedding = embeddings.embed_query(query)


# Calculate the cosine similarity between the query vector
# and every document vector.
# Higher similarity generally indicates greater semantic relevance.
scores = cosine_similarity([query_embedding], doc_embeddings)[0]


# Find the document with the highest similarity score.
# enumerate() provides both the document index and its score.
index, score = max(enumerate(scores), key=lambda x: x[1])


# Display the query, relevance score, and most semantically
# similar document.
print("query:", query)
print("similarity score:", score)
print("most similar document:", documents[index])
