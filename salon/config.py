import os

CAL_API_KEY = os.getenv("CAL_API_KEY", "")
# EU data region (api.cal.eu). Accounts created on Cal.com's European
# instance are NOT reachable via api.cal.com — that endpoint rejects the
# key with 401 "Invalid API Key". Override via CAL_API_BASE if your account
# lives on the US instance (https://api.cal.com/v2).
CAL_API_BASE = os.getenv("CAL_API_BASE", "https://api.cal.eu/v2")
DATABASE_URL = os.getenv("SALON_DATABASE_URL", "sqlite:///./salon.db")
