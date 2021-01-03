FROM python:3.8.6

ENV PYTHONUNBUFFERED 1

RUN pip install Django==3.1.2 \
                daphne==2.5.0 \
                tensorflow==2.4.0 \
                keras==2.4.3 \
                gensim==3.8.3 \
                scikit-learn==0.24.0

RUN mkdir /code
WORKDIR /code
COPY . /code

COPY docker-entrypoint.sh /usr/local/bin/
ENTRYPOINT ["bash", "docker-entrypoint.sh"]

EXPOSE 8000
CMD ["start"]