FROM python:3.8

RUN apt-get update

WORKDIR /LogisticRegression

COPY . /LogisticRegression

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

EXPOSE 5001

ENTRYPOINT ["python"]

CMD ["api.py"]