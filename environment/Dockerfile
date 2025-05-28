# hash:sha256:cedf516b759d94cec8be45c22dcfd5529a30cbf13621132b133667019f53473a
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    git+https://github.com/AllenNeuralDynamics/aind-hcr-data-transformation.git \
    aind-data-schema==1.4.0 \
    numcodecs==0.13.1 \
    numba==0.61.2 \
    tensorstore==0.1.72 \
    s3fs==2025.3.2 \
    jupyterlab
