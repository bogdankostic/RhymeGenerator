FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

RUN apt-get update
RUN apt-get -yq install openssh-server; \
    mkdir -p /var/run/sshd;

EXPOSE 22

CMD ["/usr/sbin/sshd","-D"]
