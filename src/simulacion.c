#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h> // Aseguramos incluir MPI aquí
#include "simulacion.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

// --- MACROS DE INDICES (Ajustadas a la nueva lógica) ---
// El array local tiene tamaño (iinner + 2) * (kinner + 2)
// El índice real va de 1 a iinner. 0 e iinner+1 son halos.
#define idx(i,k)  ((i) * p->kouter + (k))
#define idxn(i,k) ((i-1) * p->kinner + (k-1)) // Para el array temporal (sin halos)

// ---------------------------------------------------------
// 1. CONFIGURACIÓN DE TOPOLOGÍA (Lógica Simplificada)
// ---------------------------------------------------------
void setup_topologia(Parametros *p, int imax, int kmax, int itmax) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    p->imax = imax; p->kmax = kmax; p->itmax = itmax; p->eps = 1.e-08;

    // 1. Crear Rejilla 2D
    p->dims[0] = 0; p->dims[1] = 0;
    MPI_Dims_create(size, 2, p->dims); 
    
    int period[2] = {0, 0}; 
    MPI_Cart_create(MPI_COMM_WORLD, 2, p->dims, period, 1, &p->comm_2d);
    
    MPI_Comm_rank(p->comm_2d, &rank);
    MPI_Cart_coords(p->comm_2d, rank, 2, p->coords);
    
    MPI_Cart_shift(p->comm_2d, 0, 1, &p->top, &p->bottom);
    MPI_Cart_shift(p->comm_2d, 1, 1, &p->left, &p->right);

    // 2. División Simple (Divide y Vencerás)
    // Cada proceso recibe exactamente N / dim bloques.
    // (Asumimos que 1000 es divisible por 2. Para producción real se maneja el resto)
    p->iinner = p->imax / p->dims[0]; // Ej: 1000 / 2 = 500
    p->kinner = p->kmax / p->dims[1]; // Ej: 1000 / 2 = 500

    // Coordenadas globales de mi bloque (sin contar halos)
    p->is = p->coords[0] * p->iinner; 
    p->ks = p->coords[1] * p->kinner;
    
    // Fin de mi bloque
    p->ie = p->is + p->iinner - 1;
    p->ke = p->ks + p->kinner - 1;

    // Tamaños con halos (padding de 1 a cada lado)
    p->iouter = p->iinner + 2; 
    p->kouter = p->kinner + 2;

    // 3. Tipos de Datos (Vectores)
    // Borde Vertical: 1 columna, salto de 'kouter'
    MPI_Type_vector(p->iinner, 1, p->kouter, MPI_DOUBLE, &p->v_border_type);
    MPI_Type_commit(&p->v_border_type);

    // Borde Horizontal: Contiguo
    MPI_Type_contiguous(p->kinner, MPI_DOUBLE, &p->h_border_type);
    MPI_Type_commit(&p->h_border_type);

    if (rank == 0) {
        printf("--- Topología HPC Simplificada ---\n");
        printf("Global: %dx%d | Grid: %dx%d\n", imax, kmax, p->dims[0], p->dims[1]);
        printf("Local (Inner): %dx%d\n", p->iinner, p->kinner);
    }
}

void cleanup_mpi(Parametros *p) {
    MPI_Type_free(&p->v_border_type);
    MPI_Type_free(&p->h_border_type);
}

// ---------------------------------------------------------
// 2. ESCRITURA I/O (Modo Fila por Fila Seguro)
// ---------------------------------------------------------
void guardar_mpi_io(Parametros *p, double *phi, const char *filename) {
    MPI_File fh;
    MPI_Status status;
    int rank;
    MPI_Comm_rank(p->comm_2d, &rank);

    if (rank == 0) printf("--- Guardando %s ... ---\n", filename);

    // 1. Abrir el archivo (Esto NO borra el contenido anterior por defecto)
    int err = MPI_File_open(p->comm_2d, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) {
        if (rank == 0) printf("Error abriendo archivo MPI\n");
        return;
    }

    // TRUNCAR EL ARCHIVO A 0 BYTES 🌟
    // Esto asegura que no quede "basura" de ejecuciones anteriores
    MPI_File_set_size(fh, 0);
    
    // Barrera para asegurar que el truncado se complete antes de que nadie escriba
    MPI_Barrier(p->comm_2d); 

    // Iteramos sobre las filas LOCALES reales (índice 1 a iinner)
    for (int i = 1; i <= p->iinner; i++) {
        
        // Calcular qué fila de la matriz GLOBAL es esta
        long long global_row = p->is + (i - 1);
        long long global_col_start = p->ks;

        // Calcular OFFSET en bytes
        MPI_Offset offset = (global_row * p->kmax + global_col_start) * sizeof(double);
        
        // Escribir la fila local entera
        MPI_File_write_at(fh, offset, &phi[idx(i, 1)], p->kinner, MPI_DOUBLE, &status);
    }

    MPI_File_close(&fh);
    if (rank == 0) printf("--- Guardado Completado ---\n");
}


// ---------------------------------------------------------
// 3. SOLUCIONADOR PARALELO
// ---------------------------------------------------------
void run_2d_parallel_opt(Parametros *p) {
    // Memoria contigua con calloc (inicia en 0.0)
    double *phi = (double*)calloc(p->iouter * p->kouter, sizeof(double));
    double *phin = (double*)calloc(p->iinner * p->kinner, sizeof(double));
    
    double dx = 1.0 / p->kmax;
    double dy = 1.0 / p->imax;
    double dx2i = 1.0 / (dx*dx);
    double dy2i = 1.0 / (dy*dy);
    double dt = min(dx*dx, dy*dy) / 4.0;
    
    // --- PARÁMETROS DE FUENTES DE CALOR ---
    const int CENTER_I = p->imax / 2;
    const int CENTER_K = p->kmax / 2;
    const double TEMP_MAX = 1.0;      
    
    // Configuración de la Franja (Grosor aprox 5% de la altura total)
    // Aseguramos que tenga al menos 2 filas de grosor para que se vea bien
    const int STRIP_THICKNESS = max(2, p->imax / 20); 

    // Inicialización (Solo para ID 0 - Bordes)
    if (p->heat_source_id == 0) {
        if (p->ke >= p->kmax - 1) {
            for (int i = 1; i <= p->iinner; i++) phi[idx(i, p->kinner+1)] = TEMP_MAX;
        }
        if (p->is == 0) { 
             for (int k = 1; k <= p->kinner; k++) phi[idx(0, k)] = (p->ks + k) * dx;
        }
        if (p->ie >= p->imax - 1) { 
             for (int k = 1; k <= p->kinner; k++) phi[idx(p->iinner+1, k)] = (p->ks + k) * dx;
        }
    }

    MPI_Request reqs[8];
    MPI_Status stats[8];

    // --- BUCLE DE TIEMPO ---
    for (int it = 1; it <= p->itmax; it++) {
        
        int r_cnt = 0;
        
        // 1. Comunicación (Igual que antes)
        MPI_Irecv(&phi[idx(0, 1)], 1, p->h_border_type, p->top, 0, p->comm_2d, &reqs[r_cnt++]);
        MPI_Irecv(&phi[idx(p->iinner+1, 1)], 1, p->h_border_type, p->bottom, 1, p->comm_2d, &reqs[r_cnt++]);
        MPI_Isend(&phi[idx(1, 1)], 1, p->h_border_type, p->top, 1, p->comm_2d, &reqs[r_cnt++]);
        MPI_Isend(&phi[idx(p->iinner, 1)], 1, p->h_border_type, p->bottom, 0, p->comm_2d, &reqs[r_cnt++]);

        MPI_Irecv(&phi[idx(1, 0)], 1, p->v_border_type, p->left, 2, p->comm_2d, &reqs[r_cnt++]);
        MPI_Irecv(&phi[idx(1, p->kinner+1)], 1, p->v_border_type, p->right, 3, p->comm_2d, &reqs[r_cnt++]);
        MPI_Isend(&phi[idx(1, 1)], 1, p->v_border_type, p->left, 3, p->comm_2d, &reqs[r_cnt++]);
        MPI_Isend(&phi[idx(1, p->kinner)], 1, p->v_border_type, p->right, 2, p->comm_2d, &reqs[r_cnt++]);

        MPI_Waitall(r_cnt, reqs, stats);

        // 2. Cómputo (Igual que antes)
        double dphimax = 0.0;
        for (int i = 1; i <= p->iinner; i++) {
            for (int k = 1; k <= p->kinner; k++) {
                double term = (phi[idx(i+1,k)] + phi[idx(i-1,k)] - 2.0*phi[idx(i,k)]) * dy2i +
                              (phi[idx(i,k+1)] + phi[idx(i,k-1)] - 2.0*phi[idx(i,k)]) * dx2i;
                double dphi = term * dt;
                dphimax = max(dphimax, fabs(dphi));
                phin[idxn(i,k)] = phi[idx(i,k)] + dphi;
            }
        }
        
        // 🌟 APLICACIÓN DE FUENTES DE CALOR INTERNAS 🌟
        
// OPCIÓN 1: FUENTE CENTRAL CIRCULAR (Con diámetro)
        if (p->heat_source_id == 1) {
            
            // Radio de la fuente (ej. 5% del tamaño de la malla, o un valor fijo como 20)
            // Ajusta este valor según el grosor que desees.
            const int SOURCE_RADIUS = max(2, p->imax / 20); 
            const int RADIUS_SQ = SOURCE_RADIUS * SOURCE_RADIUS; // R al cuadrado para evitar raiz cuadrada

            for (int i = 1; i <= p->iinner; i++) {
                int i_global = p->is + (i - 1);
                
                // Optimización: Primero verificamos si estamos dentro del rango vertical del círculo
                // Si la fila está muy lejos del centro, saltamos al siguiente i
                int dist_i = i_global - CENTER_I;
                if (abs(dist_i) > SOURCE_RADIUS) continue;

                for (int k = 1; k <= p->kinner; k++) {
                    int k_global = p->ks + (k - 1);
                    int dist_k = k_global - CENTER_K;

                    // Cálculo de distancia Euclídea al cuadrado: dx^2 + dy^2 <= R^2
                    if ((dist_i * dist_i) + (dist_k * dist_k) <= RADIUS_SQ) {
                        phi[idx(i, k)] = TEMP_MAX;
                        phin[idxn(i, k)] = TEMP_MAX; 
                    }
                }
            }
        }
        
        // OPCIÓN 2: FRANJA HORIZONTAL (NUEVO)
        else if (p->heat_source_id == 2) {
            for (int i = 1; i <= p->iinner; i++) {
                int i_global = p->is + (i - 1);
                
                // Si la fila global está dentro del grosor de la franja central
                if (abs(i_global - CENTER_I) <= (STRIP_THICKNESS / 2)) {
                    // Fijamos TODA la fila a temperatura máxima
                    for (int k = 1; k <= p->kinner; k++) {
                        phi[idx(i, k)] = TEMP_MAX;
                        phin[idxn(i, k)] = TEMP_MAX; 
                    }
                }
            }
        }
        // 🌟 FIN FUENTES INTERNAS 🌟

        // 3. Update
        for (int i = 1; i <= p->iinner; i++) {
            for (int k = 1; k <= p->kinner; k++) {
                phi[idx(i,k)] = phin[idxn(i,k)];
            }
        }

        if (it % 500 == 0) {
            double global_max;
            MPI_Allreduce(&dphimax, &global_max, 1, MPI_DOUBLE, MPI_MAX, p->comm_2d);
            if (global_max < p->eps) break;
        }
    }

    // Guardar (Recuerda usar la versión corregida con MPI_File_set_size(fh, 0))
    guardar_mpi_io(p, phi, "final_1000x1000.bin");

    free(phi);
    free(phin);
}