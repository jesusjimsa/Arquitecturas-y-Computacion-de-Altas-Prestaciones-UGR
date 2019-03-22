#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char **argv){
	if(argc != 2){
		printf("Tienes que escribir el número de intervalos\n");

		exit(-1);
	}

	double width, sum_global;
	int intervals, i;
	double diferencia;
	const double PI = 3.14159265358979323846264338327950288419716939937510;
	const double ERROR = 0.50;

	int	size, rank;
	MPI_Status	status;
	
	sum_global = 0.0;

	/*
	 * Initialize MPI.
	 */
	MPI_Init(&argc, &argv);
	
	/*
	 * Error check the number of processes.
	 * Determine my rank in the world group.
	 * The sender will be rank 0 and the receiver, rank 1.
	 */
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size < 2) {
		printf("Need at least 2 processes.\n");
		MPI_Finalize();
		
		return(1);
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	/* get the number of intervals */
	intervals = atoi(argv[1]);
	width = 1.0 / intervals;

	/* do the computation */
	double sum_local = 0.0;

	for (i = rank; i < intervals; i += size){
		double x = (i + ERROR) * width;
		
		sum_local += 4.0 / (1.0 + x * x);
	}

	MPI_Reduce(&sum_local, &sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Finalize();
	
	sum_global *= width;
	diferencia = sum_global - PI;

	if(rank == 0){
		printf("Estimation of pi:\t%0.50f\n", sum_global);
		printf("Diferencia:\t\t%0.50f\n", fabs(diferencia));
	}

	return(0);
}
