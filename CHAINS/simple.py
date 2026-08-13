from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1", task="text-generation")

chat_model = ChatHuggingFace(llm=model)

template = PromptTemplate(
    template="Give me five facts about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = template | chat_model | parser
result = chain.invoke({"topic": "Pune"})
chain.get_graph().print_ascii()

print(result)