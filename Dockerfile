
FROM colmap/colmap:latest


MAINTAINER Paul-Edouard Sarlin

RUN apt-get update -y
RUN apt-get install python3 python3-pip unzip wget -y
#RUN pip3 install -U pip
COPY . /app
WORKDIR app/
RUN pip3 install -r requirements.txt
RUN pip3 install jupyterlab notebook
RUN pip3 install git+https://github.com/mihaidusmanu/pycolmap