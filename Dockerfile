FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Instala dependencias centralizadas desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/       ./src/
COPY api/       ./api/
COPY models/    ./models/
COPY reports/   ./reports/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
