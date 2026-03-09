FROM public.ecr.aws/lambda/python:3.11

# Install minimal build tools required to compile packages when wheels
# are not available on PyPI for the target Python version.
# Use microdnf which exists in the Lambda base image family.
RUN microdnf install -y gcc gcc-c++ make && microdnf clean all || true

# Use centralized requirements file so versions stay in sync across images
COPY ../requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]