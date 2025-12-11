#ifndef SIMULACION_H
#define SIMULACION_H
#include <mpi.h>

typedef struct {
    // --- Física Global ---
    int imax, kmax;     // Dimensiones totales
    int itmax;          // Iteraciones
    double eps;         // Tolerancia

    //0=Bordes (Original), 1=Centro (Puntual)
    int heat_source_id;
    
    // --- Topología MPI (Cartesiana) ---
    MPI_Comm comm_2d;       // Comunicador especial 2D
    int dims[2];            // Dimensiones de proc (ej. 2x2)
    int coords[2];          // Mis coordenadas (ej. 0,1)
    
    // Vecinos [Arriba, Abajo, Izq, Der]
    int top, bottom, left, right; 

    // --- Límites Locales (Mi pedazo de pastel) ---
    // is/ie: índices inicio/fin en Y. ks/ke: índices inicio/fin en X
    int is, ie, ks, ke;     
    int iouter, kouter;     // Tamaño total local (incluyendo halos)
    int iinner, kinner;     // Tamaño interior local (sin halos)

    // --- Tipos de Datos Derivados (Para optimizar envíos) ---
    MPI_Datatype v_border_type; // Borde Vertical (Columna)
    MPI_Datatype h_border_type; // Borde Horizontal (Fila)

} Parametros;

// Función para configurar topología y dividir el dominio
void setup_topologia(Parametros *p, int imax, int kmax, int itmax);

// Función principal del solver paralelo
void run_2d_parallel_opt(Parametros *p);

// Limpiar tipos de datos MPI
void cleanup_mpi(Parametros *p);

// Función para guardar la matriz usando escritura paralela MPI I/O
void guardar_mpi_io(Parametros *p, double *phi, const char *filename);

#endif