FROM python:3.9

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
    openssh-client \
    git \
    && apt-get clean
RUN apt-get install git-lfs -y
RUN git lfs install

RUN mkdir -p -m 0600 ~/.ssh && \
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts

WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN python cache-tokenizer.py

CMD python pipeline.py "$COMMIT_MESSAGE" "$FILENAMES"
