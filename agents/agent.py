import sqlite3
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, CHECKPOINT_DB_PATH
from tools.chem_memory import ChemicalMemoryTool
from tools.chemspider_search import ChemSpiderTool
from tools.chem_calc import ChemCalcTool


def initialize_agent():
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        temperature=0,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        extra_body={"thinking": {"type": "disabled"}},
    )

    tools = [
        ChemicalMemoryTool(),
        ChemSpiderTool(),
        ChemCalcTool(),
    ]

    system_prompt = (
        "You are ChemAssist, a professional cheminformatics assistant.\n\n"
        "You have three tools:\n"
        "1. ChemicalMemory - search historical cache for previously queried compounds.\n"
        "2. ChemSpiderSearch - query ChemSpider API for compound data (SMILES, InChIKey, MW).\n"
        "3. ChemCalc - use RDKit to compute molecular descriptors (formula, MW, LogP) and generate 2D structure images from SMILES.\n\n"
        "Core rules:\n"
        "- Always check ChemicalMemory first before calling ChemSpiderSearch.\n"
        "- After getting SMILES from ChemSpiderSearch, use ChemCalc to compute additional descriptors.\n"
        "- Never fabricate chemical structures or SMILES from memory.\n"
        "- Summarize results clearly in Chinese.\n"
        "- Inform the user when data has been cached."
    )

    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=SqliteSaver(conn),
    )
    return agent
