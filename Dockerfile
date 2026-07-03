FROM python:3.12-slim

WORKDIR /app

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

COPY salon ./salon

# SQLite lives on a mounted volume so bookings survive restarts/deploys.
ENV SALON_DATABASE_URL=sqlite:////data/salon.db
RUN mkdir -p /data
VOLUME /data

EXPOSE 8000
# Railway/Render inject PORT; default 8000 for local docker run.
CMD ["sh", "-c", "uvicorn salon.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
