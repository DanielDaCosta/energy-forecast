FROM jupyter/scipy-notebook

RUN mkdir src
WORKDIR src/
COPY requirements.txt .

RUN pip3 install -r requirements.txt
