from dotenv import load_dotenv
from agents.agent import initialize_agent


def main():
    load_dotenv()
    agent_executor = initialize_agent()

    print("🧪 ChemAssist 已启动！我是您的化学信息学助手。")
    print("您可以让我查询化学物质的SMILES、分子量等信息。输入 'quit' 退出。")

    while True:
        user_input = input("\n👤 您: ")
        if user_input.lower() == 'quit':
            print("👋 ChemAssist 已终止，再见！")
            break

        response = agent_executor.invoke({"input": user_input})
        print(f"🤖 ChemAssist: {response['output']}")


if __name__ == "__main__":
    main()