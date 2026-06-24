# Requires price data to be present before building.
# If data/input/prices_apx_2025.csv is missing, run:
#   python scripts/fetch_prices.py
# then rebuild.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry>=1.8,<3"

WORKDIR /app

# Dependency layer — cached unless pyproject.toml or poetry.lock changes
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root

COPY ev_simulator/ ./ev_simulator/
COPY app.py ./
COPY data/input/ ./data/input/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
