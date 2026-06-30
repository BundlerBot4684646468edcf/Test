import os

CAL_API_KEY = os.getenv("CAL_API_KEY", "")
CAL_API_BASE = os.getenv("CAL_API_BASE", "https://api.cal.com/v2")
DATABASE_URL = os.getenv("SALON_DATABASE_URL", "sqlite:///./salon.db")
