FROM nvcr.io/nvidia/tensorflow:21.05-tf2-py3

WORKDIR effdet

RUN python -m pip install --upgrade pip

RUN apt update

RUN python -m pip install argparse_utils

COPY . .
