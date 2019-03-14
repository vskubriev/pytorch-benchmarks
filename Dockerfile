FROM nvidia/cuda:10.0-base-ubuntu18.04
LABEL maintainer="nerox8664@gmail.com"

RUN apt update
RUN apt install -y python-pip

RUN pip install --upgrade pip
RUN pip install torch==0.4.0
RUN pip install torchvision
RUN pip install termcolor
RUN pip install attrdict
RUN pip install pyyaml

ADD test.py /root/
ADD tests.yml /root/
ADD loop.sh /root/
RUN chmod ugo+x /root/loop.sh
ENTRYPOINT python /root/test.py


