# 1. Definimos el argumento de versión
ARG PYTHON_VERSION="3.13"

# 2. Usamos la imagen base específica. 
# Nota: AWS Lambda ya provee imágenes con versiones fijas, 
# pero aquí usamos el ARG para documentar la versión.
FROM public.ecr.aws/lambda/python:${PYTHON_VERSION}

RUN echo "Usando Python version: ${PYTHON_VERSION}"

# Copiamos el archivo de versión para poder leerlo
COPY .python-version /tmp/

# Instalamos herramientas de compilación
RUN microdnf install -y gcc gcc-c++ make && microdnf clean all || true

# Copiamos los requisitos
COPY requirements.txt /tmp/requirements.txt

# Instalamos usando la versión leída del archivo y uv (altamente recomendado en lugar de pip)
# Instalar uv es extremadamente rápido y evita conflictos de resolución
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv && \
    uv pip install --system -r /tmp/requirements.txt

COPY  lambda_folder/lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]