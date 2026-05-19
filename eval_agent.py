"""
Agent evaluation script — test agent capabilities with real queries

Usage:
    .venv/Scripts/python eval_agent.py              # run all test cases
    .venv/Scripts/python eval_agent.py --mode multi # multi-agent only
    .venv/Scripts/python eval_agent.py --mode single # single-agent only
"""
import sys
import json
import time
import argparse
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Fix Windows GBK encoding for emoji
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


@dataclass
class TestCase:
    name: str
    query: str
    mode: str  # "single" or "multi"
    expect_tools: list = field(default_factory=list)
    expect_keywords: list = field(default_factory=list)
    expect_no_keywords: list = field(default_factory=list)


TEST_CASES = [
    # --- basic chem ---
    TestCase(
        name="basic-aspirin",
        query="query aspirin chemical info including SMILES and molecular weight",
        mode="multi",
        expect_tools=["ChemicalMemory", "ChemSpiderSearch"],
        expect_keywords=["SMILES", "180", "C9H8O4"],
        expect_no_keywords=["Error", "Sorry"],
    ),
    TestCase(
        name="basic-caffeine-cn",
        query="query caffeine chemical structure",
        mode="multi",
        expect_tools=["ChemicalMemory", "ChemSpiderSearch"],
        expect_keywords=["SMILES", "caffeine"],  # formula may use Unicode subscripts
        expect_no_keywords=["Error"],
    ),
    TestCase(
        name="basic-cache-hit",
        query="query aspirin again",
        mode="multi",
        expect_tools=["ChemicalMemory"],
        expect_keywords=["SMILES"],
    ),

    # --- literature ---
    TestCase(
        name="lit-AIE",
        query="search papers about AIE fluorescent molecules, at least 5",
        mode="multi",
        expect_tools=["CrossrefSearch"],
        expect_keywords=["AIE", "DOI", "10."],
        expect_no_keywords=["Error"],
    ),
    TestCase(
        name="lit-RAFT",
        query="find recent papers about RAFT polymerization chain transfer agents",
        mode="multi",
        expect_tools=["CrossrefSearch"],
        expect_keywords=["RAFT", "DOI"],
        expect_no_keywords=["Error"],
    ),

    # --- combined (both path) ---
    TestCase(
        name="combined-chem-lit",
        query="query MMA (methyl methacrylate) chemical structure and find papers about its polymerization",
        mode="multi",
        expect_tools=["ChemSpiderSearch", "CrossrefSearch"],
        expect_keywords=["SMILES", "DOI"],
        expect_no_keywords=["Error", "timeout"],
    ),

    # --- edge cases ---
    TestCase(
        name="edge-nonexistent",
        query="look up compound xyzzy12345fake",
        mode="multi",
        expect_tools=["ChemicalMemory", "ChemSpiderSearch"],
        expect_keywords=["未找到"],
    ),
    TestCase(
        name="edge-chitchat",
        query="how is the weather today?",
        mode="multi",
        expect_no_keywords=["Error", "SMILES"],
    ),

    # --- single agent ---
    TestCase(
        name="single-acetaminophen",
        query="query acetaminophen chemical info",
        mode="single",
        expect_tools=["ChemicalMemory", "ChemSpiderSearch"],
        expect_keywords=["SMILES", "151", "C8H9NO2"],
        expect_no_keywords=["Error"],
    ),
    TestCase(
        name="single-graphene-lit",
        query="search papers about graphene synthesis",
        mode="single",
        expect_tools=["CrossrefSearch"],
        expect_keywords=["graphene", "DOI"],
    ),
]


def run_test(test_case: TestCase, agent, mode: str) -> dict:
    print(f"\n{'='*60}")
    print(f"[TEST] {test_case.name}")
    print(f"  Query: {test_case.query[:80]}...")
    print(f"  Mode: {mode}")

    start = time.time()
    all_tools = set()

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=test_case.query)]},
            config={"configurable": {"thread_id": f"eval-{test_case.name}"}},
        )
        elapsed = time.time() - start

        messages = result.get("messages", [])
        output = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                output = msg.content
                break

        # count tool usage
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    all_tools.add(tc.get("name", "unknown"))

        # keyword checks
        found_keywords = [k for k in test_case.expect_keywords if k.lower() in output.lower()]
        missing_keywords = [k for k in test_case.expect_keywords if k.lower() not in output.lower()]
        unwanted_found = [k for k in test_case.expect_no_keywords if k.lower() in output.lower()]

        # tool checks
        missing_tools = [t for t in test_case.expect_tools if t not in all_tools]

        passed = (
            len(missing_keywords) == 0
            and len(unwanted_found) == 0
            and len(missing_tools) == 0
            and "Error" not in output
        )

        return {
            "name": test_case.name,
            "passed": passed,
            "elapsed": round(elapsed, 1),
            "output_preview": output[:200],
            "tools_used": sorted(all_tools),
            "found_keywords": found_keywords,
            "missing_keywords": missing_keywords,
            "unwanted_found": unwanted_found,
            "missing_tools": missing_tools,
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "name": test_case.name,
            "passed": False,
            "elapsed": round(elapsed, 1),
            "output_preview": f"EXCEPTION: {str(e)[:200]}",
            "tools_used": [],
            "found_keywords": [],
            "missing_keywords": test_case.expect_keywords,
            "unwanted_found": [],
            "missing_tools": test_case.expect_tools,
        }


def print_report(results: list):
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    total_time = sum(r["elapsed"] for r in results)

    print(f"\n{'='*70}")
    print(f"REPORT")
    print(f"{'='*70}")
    print(f"  Total: {total} | Passed: {passed} | Failed: {total - passed}")
    print(f"  Time: {total_time:.1f}s total | {total_time/total:.1f}s avg")
    print(f"{'='*70}")

    for r in results:
        status = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"\n{status} {r['name']} ({r['elapsed']}s)")
        print(f"  Output: {r['output_preview'][:120]}...")
        print(f"  Tools: {', '.join(r['tools_used']) if r['tools_used'] else '(none)'}")
        if r["missing_keywords"]:
            print(f"  MISSING KEYWORDS: {r['missing_keywords']}")
        if r["unwanted_found"]:
            print(f"  UNWANTED: {r['unwanted_found']}")
        if r["missing_tools"]:
            print(f"  MISSING TOOLS: {r['missing_tools']}")

    print(f"\n{'='*70}")
    print(f"  Pass rate: {passed}/{total} ({100*passed//total}%)")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi", "all"], default="all")
    parser.add_argument("--query", type=str, help="Run a single ad-hoc query")
    args = parser.parse_args()

    print("Initializing agents...")
    if args.mode in ("single", "all"):
        from agents.agent import initialize_agent
        single_agent = initialize_agent()
        print("  Single agent ready")
    else:
        single_agent = None

    if args.mode in ("multi", "all"):
        from agents.multiagent import initialize_multiagent
        multi_agent = initialize_multiagent()
        print("  Multi agent ready")
    else:
        multi_agent = None

    if args.query:
        agent = multi_agent or single_agent
        print(f"\nQuery: {args.query}")
        result = agent.invoke(
            {"messages": [HumanMessage(content=args.query)]},
            config={"configurable": {"thread_id": "adhoc"}},
        )
        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\nResponse:\n{msg.content}")
                break
        return

    cases = [t for t in TEST_CASES if args.mode == "all" or t.mode == args.mode]
    results = []

    for case in cases:
        agent = single_agent if case.mode == "single" else multi_agent
        if agent is None:
            print(f"  SKIP {case.name}: agent not initialized")
            continue
        result = run_test(case, agent, case.mode)
        results.append(result)

    print_report(results)


if __name__ == "__main__":
    main()
