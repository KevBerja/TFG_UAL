FROM python:3.8.1
VOLUME /TFG
WORKDIR /TFG/RandomForest
ADD . /TFG/RandomForest/
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD [ "api.py" ]