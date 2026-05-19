"""ChemAssist 集中配置模块——load_dotenv() 只在这里调用一次"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---- DeepSeek LLM ----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# ---- RSC ChemSpider ----
RSC_API_KEY = os.getenv("RSC_API_KEY")
RSC_BASE_URL = "https://api.rsc.org/compounds/v1"

# ---- CrossRef ----
CROSSREF_BASE_URL = "https://api.crossref.org/works/"
CROSSREF_TIMEOUT = 15
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "user@example.com")

# ---- 会话持久化 ----
CHECKPOINT_DB_PATH = os.environ.get(
    "CHEMASSIST_TEST_DB",
    os.path.join(os.path.dirname(__file__), "local_data", "checkpoints.db"),
)
