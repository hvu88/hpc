#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "simulacion.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // Argumentos
    int imax = (argc > 1) ? atoi(argv[1]) : 200;
    int kmax = (argc > 2) ? atoi(argv[2]) : 200;
    int it   = (argc > 3) ? atoi(argv[3]) : 50000;
    int heat_source_id = (argc > 4) ? atoi(argv[4]) : 0; // 0 por defecto (Bordes

    Parametros p;
    //Asigna el ID de la fuente a la estructura de parámetros.
    p.heat_source_id = heat_source_id;

    // La magia de configuración ocurre aquí dentro
    setup_topologia(&p, imax, kmax, it);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    run_2d_parallel_opt(&p);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        printf("Tiempo Final (2D MPI): %f segundos\n", end - start);
    }

    cleanup_mpi(&p);
    MPI_Finalize();
    return 0;
}