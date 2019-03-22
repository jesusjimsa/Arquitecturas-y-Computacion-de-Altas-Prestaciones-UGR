#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv){
	if(argc != 3){
		printf("Tienes que escribir dos argumentos: número de intervalos y error\n");
		printf("Ejemplo: ./cpi-seq 1000 0.5\n");

		exit(-1);
	}

	register double width, sum;
	register int intervals, i;
	register double diferencia;
	register double error;
	const double PI = 3.14159265358979323846264338327950288419716939937510;
	
	/* get the number of intervals */
	intervals = atoi(argv[1]);
	error = atof(argv[2]);
	width = 1.0 / intervals;

	/* do the computation */
	sum = 0;
	
	for (i = 0; i < intervals; ++i) {
		register double x = (i + error) * width;
		
		sum += 4.0 / (1.0 + x * x);
	}
	
	sum *= width;
	diferencia = sum - PI;

	printf("Estimation of pi is\t%0.50f\n", sum);
	printf("Diferencia:\t\t%0.50f\n", fabs(diferencia));

	return(0);
}

