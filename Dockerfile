FROM ubuntu:22.04

RUN apt-get update\
    && apt-get install -y \
        nano \
        python3 \
        python3-pip

RUN mkdir SafeML

ADD SafeML SafeML

RUN pip install -r SafeML/requirements.txt
