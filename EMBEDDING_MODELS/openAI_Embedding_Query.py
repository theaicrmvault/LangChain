from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=32)

vector = embeddings.embed_query("Explain the theory of relativity in simple terms")
print(str(vector))