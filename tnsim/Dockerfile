FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/tnsim
RUN useradd --create-home --uid 10001 tnsim && mkdir /app/cache && chown tnsim:tnsim /app/cache
USER tnsim
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "tnsim.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
