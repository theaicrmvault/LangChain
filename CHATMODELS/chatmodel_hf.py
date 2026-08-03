from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
                          repo_id="Qwen/Qwen2.5-72B-Instruct",
                          task="text-generation"
                          )

chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke('generate acls poem')
print(result.content)