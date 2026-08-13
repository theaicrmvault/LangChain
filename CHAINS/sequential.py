from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1", task="text-generation")
chat_model = ChatHuggingFace(llm=model)

template1 = PromptTemplate(
    template="Write a article on {topic} in 500 words",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Give main points in \n Article: {article}",
    input_variables=["article"]
)

parser = StrOutputParser()

chain = template1 | chat_model | template2 | chat_model | parser
result = chain.invoke({"topic": "KV caching in LLMs"})

print(result)
