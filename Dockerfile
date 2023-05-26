FROM python:3.9.16-slim-bullseye

WORKDIR /app
COPY ./models /app/models
COPY ./cache-tokenizer.py /app/cache-tokenizer.py
COPY ./pipeline.py /app/pipeline.py
COPY ./requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt \
    && python cache-tokenizer.py

CMD python pipeline.py "$COMMIT_MESSAGE" "$FILENAMES"
