# hash:sha256:50f80eb345256aeff785c23e9229271578e805c60efff3b041aa803e1d452dcd
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    git+https://github.com/AllenNeuralDynamics/aind-hcr-data-transformation.git@dev \
    aind-data-schema==1.4.0 \
    numba==0.61.2 \
    jupyterlab
