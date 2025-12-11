# Usamos una imagen base de Linux (Ubuntu)
FROM ubuntu:22.04

# Evitamos preguntas interactivas durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Actualizamos e instalamos herramientas base, MPI y Python
RUN apt-get update && apt-get install -y \
    build-essential \
    openmpi-bin \
    libopenmpi-dev \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Establecemos el directorio de trabajo
WORKDIR /workspace

# Instalamos dependencias de Python
RUN pip3 install --no-cache-dir streamlit matplotlib numpy plotly pandas

# Copiamos TODO
COPY . .

# Compilar
RUN make

# Exponer el puerto
EXPOSE 8501

# Comando de arranque
# Usamos "python3 -m streamlit" que es más seguro para encontrar el ejecutable
CMD ["python3", "-m", "streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]