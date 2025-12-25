FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

COPY requirements2.txt .

COPY requirements3.txt .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements2.txt --no-cache-dir
RUN pip install -r requirements3.txt --no-cache-dir


WORKDIR /app

RUN chmod -R a+rX /app

COPY . .

RUN pip install .

EXPOSE 8888

CMD ["pwd"]
CMD ["ls"]

RUN pip uninstall -y watchdog

ENV STREAMLIT_WATCH_FILE_CHANGES=false
ENV STREAMLIT_SERVER_FILEWATCHERTYPE=none
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_GLOBAL_DISABLEWATCHDOGWARNING=true

ENV PYTHONPATH="/app:${PYTHONPATH}"

# For UI
CMD ["streamlit", "run", "/app/main.py", "--server.port=8888"]
