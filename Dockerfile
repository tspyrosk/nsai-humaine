FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY requirements2.txt .
COPY requirements3.txt .

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements2.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements3.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install jupyterlab nbformat ipykernel ipywidgets
RUN python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3"
RUN pip uninstall -y watchdog

COPY . .

RUN pip install .

RUN mkdir -p /app/input /app/output /app/notebooks

COPY nginx.conf /etc/nginx/nginx.conf
COPY supervisord.conf /etc/supervisor/supervisord.conf

ENV PYTHONPATH="/app"
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_FILEWATCHERTYPE=none
ENV STREAMLIT_GLOBAL_DISABLEWATCHDOGWARNING=true

EXPOSE 8888

CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
