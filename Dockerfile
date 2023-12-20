FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN python -m pip install --upgrade pip==20.3.4
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

ENTRYPOINT ["python", "app.py"]