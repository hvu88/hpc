# Usamos una imagen base de Linux (Ubuntu 22.04)
FROM ubuntu:22.04

# Variables de entorno para evitar preguntas en la instalación y mejorar logs en Render
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Actualizamos e instalamos dependencias del sistema y MPI
RUN apt-get update && apt-get install -y \
    build-essential \
    openmpi-bin \
    libopenmpi-dev \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Establecemos el directorio de trabajo
WORKDIR /app

# Instalamos las librerías de Python
# Nota: --no-cache-dir reduce el tamaño de la imagen final
RUN pip3 install --no-cache-dir streamlit matplotlib numpy plotly

# 🌟 PASO CRÍTICO: Copiar los archivos del proyecto 🌟
# Copiamos todo el contenido de tu repositorio al contenedor
COPY . .

# 🌟 COMPILACIÓN MPI 🌟
# Si tienes un Makefile, esto compilará tu código C automáticamente.
# El 'if' evita que falle si por alguna razón no hay Makefile.
RUN if [ -f Makefile ]; then make; fi

# Koyeb inyectará la variable PORT (por defecto 8000)
EXPOSE 8000

# Usamos sh -c para capturar la variable de entorno
CMD sh -c "streamlit run dashboard.py --server.port=${PORT:-8000} --server.address=0.0.0.0"