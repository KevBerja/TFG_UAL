FROM python:3.8

RUN apt-get update

WORKDIR /RandomForest

COPY . /RandomForest

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

EXPOSE 5002

ENTRYPOINT ["python"]

CMD ["api.py"]