FROM python:3.7.0

RUN pip install mlflow==1.0.0
RUN pip install psycopg2

RUN mkdir -p /server
WORKDIR /server
COPY . /server

RUN chmod +x wait-for-it.sh

#might not need that since artifacts go to the client
#RUN mkdir /mlflow/

CMD ./wait-for-it.sh postgres:5432 -- mlflow server \
    --backend-store-uri postgresql://postgres:admin@postgres:5432/mlflow \
    --default-artifact-root /mlflow \
    --host 0.0.0.0