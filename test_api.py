import os, requests, time
from dotenv import load_dotenv

load_dotenv()
HEADERS = {"apikey": os.getenv("RSC_API_KEY")}
BASE = "https://api.rsc.org/compounds/v1"

# 步骤 1：用名称发起查询
resp = requests.post(f"{BASE}/filter/name",
                      json={"name": "aspirin"},
                      headers=HEADERS)
resp.raise_for_status()
print("1. 提交结果:", resp.json())

# 步骤 2：轮询状态
query_id = resp.json().get("queryId")
for i in range(15):
    time.sleep(2)
    s = requests.get(f"{BASE}/filter/{query_id}/status", headers=HEADERS)
    s.raise_for_status()
    print(f"2. 状态检查 {i+1}:", s.json())
    if s.json().get("status") in ["Complete", "Failed", "Suspended", "NotFound"]:
        break

# 步骤 3：获取结果
r = requests.get(f"{BASE}/filter/{query_id}/results", headers=HEADERS)
r.raise_for_status()
print("3. 结果数组:", r.json())

# 步骤 4：如果结果非空，取第一条 ID 查详情
results = r.json().get("results", [])
if results:
    # 注意：文档说 results 是 "Record IDs" 数组，可能是 int 也可能是 dict
    first = results[0]
    cid = first if isinstance(first, int) else first.get("id")
    detail = requests.get(f"{BASE}/records/{cid}/details",
                          params={"fields": "SMILES,Formula,StdInChIKey"},
                          headers=HEADERS)
    detail.raise_for_status()
    print("4. 详情:", detail.json())
else:
    print("4. 结果为空 → API 确实无法通过名称找到苯甲酸")