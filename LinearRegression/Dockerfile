FROM python:3.8

WORKDIR /LinearRegression

COPY . /LinearRegression

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python"]

CMD ["api.py"]