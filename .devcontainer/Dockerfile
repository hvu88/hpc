FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    openmpi-bin \
    libopenmpi-dev \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir streamlit matplotlib numpy plotly

# Copiamos todo
COPY . .

# 🕵️‍♂️ DIAGNÓSTICO: Esto mostrará en los logs qué archivos llegaron realmente
RUN echo "============ ARCHIVOS EN /app ============" && \
    ls -laR /app && \
    echo "=========================================="

RUN if [ -f Makefile ]; then make; fi

EXPOSE 8501

# Usamos ruta relativa por seguridad, asumiendo que estamos en /app
CMD sh -c "streamlit run dashboard.py --server.port=${PORT:-8501} --server.address=0.0.0.0"