FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN python -m pip install --upgrade pip==23.3.2
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "main.py"]