FROM python:3.8.1
VOLUME /TFG
WORKDIR /TFG/NaturalLanguageToolkit
ADD . /TFG/NaturalLanguageToolkit/
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD [ "api.py" ]