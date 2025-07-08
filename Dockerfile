FROM python:3.13-slim
VOLUME ["/data"]
WORKDIR /app
COPY src/ ./src/
COPY pyproject.toml ./
ENV PYTHONPATH=/app
RUN pip install uv && uv pip install --no-cache --system .
CMD ["python", "src/bot.py"]