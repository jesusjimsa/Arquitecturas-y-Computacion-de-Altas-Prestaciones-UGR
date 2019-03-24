#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

int main(int argc, char **argv){
	if(argc != 2){
		printf("Tienes que escribir el número de intervalos\n");

		exit(-1);
	}

	register double width, sum;
	register int intervals, i;
	register double diferencia;
	const double PI = 3.14159265358979323846264338327950288419716939937510;
	clock_t start, end;
	double cpu_time_used;

	start = clock();

	/* get the number of intervals */
	intervals = atoi(argv[1]);
	width = 1.0 / intervals;

	/* do the computation */
	sum = 0;

	for (i = 0; i < intervals; ++i) {
		register double x = (i + 0.5) * width;

		sum += 4.0 / (1.0 + x * x);
	}

	sum *= width;
	diferencia = sum - PI;

	end = clock();

	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("Estimation of pi:\t%0.50f\n", sum);
	printf("Diferencia:\t\t%0.50f\n", fabs(diferencia));
	printf("Tiempo en 1 máquina:\t%0.50f\n", cpu_time_used);

	return(0);
}

