FROM public.ecr.aws/lambda/python:3.12

# Instala dependencias
RUN pip install --no-cache-dir \
    xgboost \
    scikit-learn \
    pandas \
    numpy \
    joblib

# Copia el handler
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Handler
CMD ["lambda_function.lambda_handler"]