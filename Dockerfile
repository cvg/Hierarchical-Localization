FROM colmap/colmap:latest
MAINTAINER Paul-Edouard Sarlin
RUN apt-get update -y && apt-get install -y unzip wget python3-pip
COPY . /app
WORKDIR app/
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade notebook ipywidgets
