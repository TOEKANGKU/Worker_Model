FROM python:3-alpine

RUN apk update && apk upgrade
RUN apk add git

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod 444 app.py
RUN chmod 444 requirements.txt

ENV PORT 8080

CMD [ "python", "app.py" ]