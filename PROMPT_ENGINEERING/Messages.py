from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
history = []
history.append(SystemMessage(content='You are a helpful assistant.'))
while True:
    input_text = input("You: ")
    if(input_text.lower() == 'exit'):
        break
    history.append(HumanMessage(content=input_text))
    result = model.invoke(history);
    print("AI: " + result.content)
    history.append(AIMessage(content=result.content))





