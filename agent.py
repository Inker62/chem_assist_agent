import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools.chemspider_search import ChemSpiderTool

load_dotenv()

def initialize_agent():
    llm = ChatOpenAI(
        model="deepseek-v4-flash",  # 或者 "deepseek-chat"
        temperature=0,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"), # 从环境变量读取
        openai_api_base="https://api.deepseek.com",
        extra_body={"thinking":{"type":"disabled"}}
    )

    # 加载工具库
    tools = [ChemSpiderTool()]

    # 为Agent编写“科学家”人设提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的化学信息学助手，名叫“ChemAssist”。
            你的核心任务是：
            1. 根据用户的自然语言指令，调用工具完成任务。
            2. 如果工具返回了化学数据，请用清晰、专业的中文总结关键信息（如SMILES, 分子量）。
            3. 如果工具返回错误，如实告知用户并尝试给出建议。
            不要在未调用工具的情况下，凭记忆编造任何化学结构数据
            你绝对不能在任何情况下凭记忆或常识输出 SMILES、分子式等化学结构数据。"""),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # 用于存放思考和工具调用记录
    ])

    # 组装Agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True让思考过程在终端可见