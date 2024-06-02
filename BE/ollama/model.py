from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3", format="json", temperature=0)
