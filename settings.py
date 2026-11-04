from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PDF = ROOT / "data" / "Mind Management, Not Time Management.pdf"
DB_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "mind_management"
