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

# Copiamos solo requisitos de inferencia para Lambda
COPY lambda_folder/requirements.lambda.txt /tmp/requirements.lambda.txt

# Instalamos solo dependencias runtime mínimas para reducir tamaño/latencia
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv && \
    uv pip install --system -r /tmp/requirements.lambda.txt && \
    pip cache purge

# Copiar artefactos preentrenados (modelo JSON + scaler) para carga sin S3
# Esto hace que el cold start sea instantáneo (no descarga desde S3)
RUN mkdir -p ${LAMBDA_TASK_ROOT}/models
COPY models/model.json ${LAMBDA_TASK_ROOT}/models/model.json
COPY models/scaler.pkl ${LAMBDA_TASK_ROOT}/models/scaler.pkl

COPY lambda_folder/lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]