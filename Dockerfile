FROM python:3.10

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./API/main_tf-serving.py /app/main_tf-serving.py

CMD ["uvicorn", "main_tf-serving:app", "--host", "0.0.0.0", "--port", "80"]
