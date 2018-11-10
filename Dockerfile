FROM nvidia/cuda:9.0-base-ubuntu16.04
LABEL maintainer="nerox8664@gmail.com"

RUN apt update
RUN apt install -y python-pip

RUN pip install torch==0.4.0
RUN pip install torchvision
RUN pip install termcolor
RUN pip install attrdict
RUN pip install pyyaml

ADD test.py /root/
ENTRYPOINT python /root/test.py


