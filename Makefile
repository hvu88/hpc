CC = mpicc
# AQUI AGREGAMOS -O3 (Optimización máxima) y -lm (Librería matemática)
CFLAGS = -Wall -O3
LDFLAGS = -lm 
TARGET = simulacion_app

SRCS = main.c simulacion.c

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)