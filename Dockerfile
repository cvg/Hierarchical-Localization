FROM colmap/colmap:latest
MAINTAINER Paul-Edouard Sarlin
RUN apt-get update -y
RUN apt-get install python3 python3-pip unzip wget -y
COPY . /app
WORKDIR app/
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install jupyterlab notebook
