# Usamos una imagen base de Linux (Ubuntu)
FROM ubuntu:22.04

# Evitamos preguntas interactivas durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Actualizamos el sistema e instalamos las herramientas base, MPI y Python
RUN apt-get update && apt-get install -y \
    build-essential \
    openmpi-bin \
    libopenmpi-dev \
    git \
    gdb \
    # 🌟 INICIO: Se añaden los paquetes de Python y pip 🌟
    python3 \
    python3-pip \
    # 🌟 FIN: Se añaden los paquetes de Python y pip 🌟
    && rm -rf /var/lib/apt/lists/*

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /workspace

# 🌟 PASO ADICIONAL: Instalación de las librerías de Python 🌟
# Aquí instalamos las bibliotecas que necesita tu dashboard (Streamlit y Matplotlib)
RUN pip3 install streamlit matplotlib numpy plotly

# Ahora puedes seguir con los pasos específicos de tu proyecto:
# COPY . .
# RUN make
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]