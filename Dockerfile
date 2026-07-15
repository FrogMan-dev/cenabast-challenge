FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-dev.txt requirements-test.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Entrenamiento en build-time: genera challenge/artifacts/model.pkl
# dentro de la imagen, para que la API arranque ya con el modelo cargado.
RUN python -m challenge.train

EXPOSE 8080
CMD ["uvicorn", "challenge:app", "--host", "0.0.0.0", "--port", "8080"]
