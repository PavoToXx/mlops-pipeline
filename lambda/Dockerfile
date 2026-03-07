FROM public.ecr.aws/lambda/python:3.11

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "numpy==1.26.4" \
        "scipy==1.13.1" \
        "scikit-learn==1.6.1" \
        "xgboost==2.1.3" \
        "pandas==2.2.3" \
        "joblib==1.4.2"

COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]