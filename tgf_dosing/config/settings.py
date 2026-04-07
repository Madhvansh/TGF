"""
TGF Settings - Environment variable overrides for configuration.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.environ.get("TGF_DATA_PATH", os.path.join(PROJECT_ROOT, "data", "Parameters_5K.csv"))
DB_PATH = os.environ.get("TGF_DB_PATH", "tgf_data.db")
MOMENT_CHECKPOINT = os.environ.get("TGF_MOMENT_CHECKPOINT", None)
API_PORT = int(os.environ.get("TGF_API_PORT", "8000"))
LOG_LEVEL = os.environ.get("TGF_LOG_LEVEL", "INFO")
