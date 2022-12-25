FROM python:3.10

WORKDIR /app
COPY . .
RUN pip install poetry
RUN poetry install
RUN apt install tk
CMD poetry shell; poetry run python main.py -c feed_forward/image_classification/manual/n_layer/n_layer.yaml
