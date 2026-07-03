import os

DATABASE_URL = os.getenv("SALON_DATABASE_URL", "sqlite:///./salon.db")

# Neon / Vercel Postgres hand out "postgres://..." URLs; SQLAlchemy expects
# the "postgresql://" scheme (psycopg2 driver, installed via requirements).
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]
