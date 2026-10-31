# The visualizer: the browser app, built by Vite, served by the FastAPI
# server along with its API. Terraform builds this (`wildhat admin
# build-image visualizer`) whenever a file in this folder changes.

FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.13-slim
WORKDIR /app
COPY server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY server/app.py ./
COPY --from=web /web/dist ./web
ENV WEB_DIR=/app/web PYTHONUNBUFFERED=1
# Cloud Run sets PORT.
CMD ["sh", "-c", "exec uvicorn app:main --factory --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers"]
