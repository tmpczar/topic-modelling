FROM python:3.11-slim

ENV BASE_PATH /topic-modelling


COPY . /topic-modelling/

WORKDIR /topic-modelling

ENV PYTHONPATH "${PYTHONPATH}:/topic-modelling/src"

# requires gcc for hdbscan
RUN apt-get update && \
    apt-get install -y gcc build-essential libpython3-dev


RUN pip install --no-cache-dir -r requirements.txt

RUN python /topic-modelling/src/nltk_setup.py

RUN python /topic-modelling/src/train.py

RUN chmod +x ./entrypoint.sh

ENTRYPOINT [ "./entrypoint.sh" ]

