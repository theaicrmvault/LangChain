from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1", task="text-generation")
chat_model = ChatHuggingFace(llm=model)

template1 = PromptTemplate(
    template="Write a article on {text} in 500 words", 
    input_variables=["text"]
)
template2 = PromptTemplate(
    template="Summarize the following text in 100 words: {text}",
    input_variables=["text"],
)
template3 = PromptTemplate(
    template="Create mnemonics for the \n Article: {text}", 
    input_variables=["text"]
)
template4 = PromptTemplate(
    template="merge summary \n {summarize_chain} and mnemonics \n {mnemonics_chain}", 
    input_variables=["summarize_chain","mnemonics_chain"]
)

parser = StrOutputParser()

chain1 = template1 | chat_model | parser

parellel_chain = RunnableParallel(
    {
        "summarize_chain": template2 | chat_model | parser,
        "mnemonics_chain": template3 | chat_model | parser,
    }
)
merged_chain = template4 | chat_model | parser

final_chain = chain1 | parellel_chain | merged_chain

final_chain.get_graph().print_ascii()

result = final_chain.invoke({"text": "KV caching in LLMs"})
print(result)
