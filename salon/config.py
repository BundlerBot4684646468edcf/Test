import os

DATABASE_URL = os.getenv("SALON_DATABASE_URL", "sqlite:///./salon.db")
