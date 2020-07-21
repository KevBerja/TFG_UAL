FROM python:3.8

RUN apt-get update

WORKDIR /LinearRegression

COPY . /LinearRegression

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python"]

CMD ["api.py"]