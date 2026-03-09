# Definimos el argumento con un valor por defecto
ARG PYTHON_VER = "3.13"


# Usamos la variable para traer la imagen
FROM python:${PYTHON_VER}-slim

RUN ECHO"Usando Python version: ${PYTHON_VER}"

# Instalamos uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copiamos la definición de versión al contenedor
COPY .python-version /app/
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
