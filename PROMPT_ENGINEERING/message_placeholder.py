from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

template = ChatPromptTemplate(
    [('system', 'You are a helpful assistant.'),
     MessagesPlaceholder(variable_name='stored_messages'),
     ('human', 'tell me about {input_text}')])

chat_history =[]
with open('stored_messages.txt') as f:
    chat_history.extend(f.readlines())
    
result = template.invoke({'input_text': 'Tell me example again', 'stored_messages': chat_history})

print(result)