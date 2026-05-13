import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI


from tools.chem_memory import ChemicalMemoryTool
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
    tools = [
        ChemicalMemoryTool(),
        ChemSpiderTool()
    ]

    # 为Agent编写“科学家”人设提示词
    system_prompt =  """你是一个专业的化学信息学助手，名叫“ChemAssist”。
            你有两个工具可以调度：
            1.ChemicalMemory：用于搜索之前曾查询过的历史缓存。每次收到用户请求时，必须使用此工具；
            2.ChemSpiderSearch：当ChemicalMemory返回'Not Found'结果时，才使用此工具从ChemSpider API查询化学物质信息。
            
            你的核心任务是：
            1. 根据用户的自然语言指令，调用工具完成任务。必须先调用ChemicalMemory查询历史缓存。若缓存命中，直接返回其中的化学信息。若返回'Not Found'，再调用ChemSpiderSearch获取新数据
            2. 如果ChemSpiderSearch工具返回了化学数据，请用清晰、专业的中文总结关键信息（如SMILES, 分子量）。
            3. 如果ChemSpiderSearch工具返回错误，如实告知用户并尝试给出建议。
            4. 当调用ChemSpiderSearch工具获取了新数据，请告知用户'已为您缓存此数据'
            不要在未调用工具的情况下，凭记忆编造任何化学结构数据
            你绝对不能在任何情况下凭记忆或常识输出 SMILES、分子式等化学结构数据。"""

    # 组装Agent
    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt, checkpointer=InMemorySaver())
    return agent
