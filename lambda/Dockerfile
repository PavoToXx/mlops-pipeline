FROM public.ecr.aws/lambda/python:3.11

RUN pip install --no-cache-dir \
    xgboost==2.1.3 \
    scikit-learn==1.6.1 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    joblib==1.4.2

COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]