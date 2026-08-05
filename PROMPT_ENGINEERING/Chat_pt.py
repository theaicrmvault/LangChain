from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate(
    [('system', 'You are a helpful assistant.'),
    ('human', 'tell me about {input_text}')]
)

result = template.invoke({'input_text': 'Prompt Caching'})
print(result)


