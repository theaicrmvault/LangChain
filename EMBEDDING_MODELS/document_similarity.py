from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=300)

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
    "My laptop battery drains quickly when running machine learning models locally."
]

query = "Tell me about the Eiffel Tower and its history."

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = max(enumerate(scores), key=lambda x: x[1])

print('query:', query)
print('similarity score', score)
print('most similar document:',documents[index])