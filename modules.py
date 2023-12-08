from langchain.agents import AgentType, Tool, initialize_agent
#from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import MessagesPlaceholder
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
#from langchain.schema import HumanMessage, AIMessage



# agent = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.OPENAI_FUNCTIONS,
#     verbose=True,
#     agent_kwargs=agent_kwargs,
#     memory=memory,
#     max_iterations=2
# )

def runner(query, messages):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",openai_api_key="add your key")
    db = SQLDatabase.from_uri("sqlite:///apartment_listings.db")
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    tools = [
        # Tool(
        #     name="Search",
        #     func=search.run,
        #     description="useful for when you need to answer questions about current events. You should ask targeted questions",
        # ),
        # Tool(
        #     name="Calculator",
        #     func=llm_math_chain.run,
        #     description="useful for when you need to answer questions about math",
        # ),
        Tool(
            name="Apartments-DB",
            func=db_chain.run,
            description="Useful when asking any questions related to real estate. Input to this tool must Strictly be a SINGLE JSON STRING",
        ),
    ]

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
    }
    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retrieved_chat_history = ChatMessageHistory(messages=messages)
    # retrieved_chat_history.add_user_message("Tell me the cheapest place")

    # retrieved_chat_history.add_ai_message("Jayanagar")
    retrieved_memory = ConversationBufferWindowMemory(chat_memory=retrieved_chat_history,memory_key='chat_history', return_messages=True, k=5 )
    agent = initialize_agent(
                        tools,
                        llm,
                        agent=AgentType.OPENAI_FUNCTIONS,
                        verbose=True,
                        agent_kwargs=agent_kwargs,
                        memory=retrieved_memory,
                        max_iterations=2
                    )
    result = agent.run(query)
    extracted_messages = agent.memory.chat_memory.messages
    #print(result)
    return result, extracted_messages

if __name__ == '__main__':
    messages = []#[HumanMessage(content='which place has cheapest options'), AIMessage(content='Jayanagar is the cheapest.')]
    while True:
        inp = input("Enter a message: ")
        if inp == 'bye':
            break
        result,hist = runner(inp,messages)
        messages.extend(hist)
        print(f'{result}')#\n\n{messages}')
    # result, hist = runner("What is the place?",messages)
    # messages.extend(hist)
    # print(f'{result}\n\n{messages}')
    # result = agent.run("which place has cheapest options")
    # #result2 = agent.run("what is the best price I can get there")
    # extracted_messages = agent.memory.chat_memory.messages
    # # ingest_to_db = messages_to_dict(extracted_messages)
    # # retrieve_from_db = json.loads(json.dumps(ingest_to_db))
    # # retrieved_messages = messages_from_dict(retrieve_from_db)
    # # retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
    # # retrieved_memory = ConversationBufferMemory(chat_memory=retrieved_chat_history)
    # print(f"{result}\n\n\n\n{extracted_messages}")