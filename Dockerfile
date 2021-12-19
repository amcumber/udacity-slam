FROM nvidia/cuda:11.0-runtime AS jupyter-base
LABEL citation="https://towardsdatascience.com/deep-learning-with-containers-part-1-4779877492a1"

# LABEL author="aaron.mcumber@gmail.com"
WORKDIR /
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools
RUN pip3 -q install pip --upgrade
RUN pip3 install \
    jupyter \
    numpy pandas \
    torch torchvision \
    tensorboardX

FROM jupyter-base
# RUN pip3 install \
#     transformers \
#     barbar

RUN mkdir /workspace
RUN mkdir -p /data && \
    cd /workspace

WORKDIR /workspace/

COPY . .

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", \
    "--allow-root"]
